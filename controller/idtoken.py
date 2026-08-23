"""Reading an ID token, and refusing to believe one that is not signed.

This is the only place in the studio where a *third party* gets to say who
somebody is. Everything downstream -- provisioning an account, matching one by
address, granting administrator -- trusts the dictionary that comes out of
`verify()`. So the checks are done here, in one function, rather than being
spread across the caller where one of them can quietly go missing.

A word on why this file exists at all rather than a dependency. The controller
deliberately carries no GPU stack and few libraries, and it already depends on
`cryptography` for encrypting the tokens people entrust to it -- which is the
hard part of signature verification. What remains is base64url, a JSON parse,
and the five claim checks that OpenID Connect spells out. That is small enough
to read in full, which is the property that matters most for the one piece of
code that decides who you are.

What is checked, and why each one is not optional:

* **the signature**, against the provider's published key, matched by `kid`.
  Without it an ID token is a note from the browser rather than from the
  provider, and the browser is the thing being authenticated.
* **`iss`** exactly equals the issuer we discovered. A valid token from a
  different provider is still somebody else's token.
* **`aud`** contains our client id. A token minted for another application at
  the same provider is not ours to accept -- this is the confused-deputy case,
  and it is the one people forget.
* **`exp`/`iat`**, with a small clock skew allowance, because two servers
  never agree to the second and refusing a token issued half a second in the
  future is a support ticket rather than a security win.
* **`nonce`** equals the one we generated for this sign-in. This is what makes
  a stolen-and-replayed token useless.

Unsupported algorithms are refused rather than skipped. `alg: none` is the
oldest hole in JWT and it is closed here by the simple fact that "none" is not
in the table of things this file knows how to verify.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx

# Fetched keys, per JWKS url. Providers rotate signing keys, so this is a
# cache with a lifetime rather than a one-time load -- but a token naming a
# `kid` we have not seen also forces a refresh, which is what actually makes
# rotation invisible.
_JWKS: dict[str, tuple[float, dict]] = {}
_JWKS_TTL_S = 3600.0
_MIN_REFETCH_S = 30.0
_last_fetch: dict[str, float] = {}

LEEWAY_S = 120.0


class TokenError(Exception):
    """A token that will not be believed, with the reason in plain words."""


def b64url(data: str) -> bytes:
    """Decode base64url, restoring the padding JWT leaves off."""
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _segments(token: str) -> tuple[dict, dict, bytes, bytes]:
    try:
        head_b64, body_b64, sig_b64 = token.split(".")
        header = json.loads(b64url(head_b64))
        payload = json.loads(b64url(body_b64))
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise TokenError("The provider returned something that is not an "
                         "ID token.") from e
    return header, payload, b64url(sig_b64), ("%s.%s" % (head_b64, body_b64)).encode()


async def _jwks(url: str, force: bool = False) -> dict:
    now = time.time()
    cached = _JWKS.get(url)
    if cached and not force and now - cached[0] < _JWKS_TTL_S:
        return cached[1]
    # A token naming an unknown key forces a refetch, so an attacker sending
    # nonsense `kid`s must not be able to turn that into a request per attempt.
    if force and now - _last_fetch.get(url, 0.0) < _MIN_REFETCH_S:
        if cached:
            return cached[1]
    _last_fetch[url] = now
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    _JWKS[url] = (now, data)
    return data


def _public_key(jwk: dict):
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    kty = jwk.get("kty")
    if kty == "RSA":
        n = int.from_bytes(b64url(jwk["n"]), "big")
        e = int.from_bytes(b64url(jwk["e"]), "big")
        return rsa.RSAPublicNumbers(e, n).public_key()
    if kty == "EC":
        curves = {"P-256": ec.SECP256R1(), "P-384": ec.SECP384R1(),
                  "P-521": ec.SECP521R1()}
        curve = curves.get(jwk.get("crv"))
        if curve is None:
            raise TokenError("The provider signs with a curve this studio "
                             "does not support (%s)." % jwk.get("crv"))
        return ec.EllipticCurvePublicNumbers(
            int.from_bytes(b64url(jwk["x"]), "big"),
            int.from_bytes(b64url(jwk["y"]), "big"), curve).public_key()
    raise TokenError("The provider signs with a key type this studio does not "
                     "support (%s)." % kty)


def _check_signature(alg: str, key, signed: bytes, signature: bytes) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, utils

    digests = {"256": hashes.SHA256(), "384": hashes.SHA384(),
               "512": hashes.SHA512()}
    digest = digests.get(alg[2:])
    if digest is None or alg[:2] not in ("RS", "PS", "ES"):
        # "none" lands here, as does anything else unrecognised. Refusing by
        # default is the whole point: a new algorithm should fail closed until
        # somebody adds it deliberately.
        raise TokenError("The provider signed with %s, which this studio does "
                         "not verify. Ask it for RS256." % alg)
    try:
        if alg.startswith("RS"):
            key.verify(signature, signed, padding.PKCS1v15(), digest)
        elif alg.startswith("PS"):
            key.verify(signature, signed,
                       padding.PSS(mgf=padding.MGF1(digest),
                                   salt_length=digest.digest_size), digest)
        else:
            # ECDSA in a JWT is the raw r||s pair, not the DER structure
            # `cryptography` expects -- so it is converted rather than passed
            # through, which would fail on every valid token.
            half = len(signature) // 2
            der = utils.encode_dss_signature(
                int.from_bytes(signature[:half], "big"),
                int.from_bytes(signature[half:], "big"))
            key.verify(der, signed, ec.ECDSA(digest))
    except InvalidSignature as e:
        raise TokenError("The ID token signature does not match the "
                         "provider's published key.") from e


async def verify(token: str, jwks_uri: str, issuer: str, audience: str,
                 nonce: str | None = None) -> dict[str, Any]:
    """The claims in this token, or TokenError explaining why not."""
    header, payload, signature, signed = _segments(token)

    kid = header.get("kid")
    keys = (await _jwks(jwks_uri)).get("keys") or []
    match = next((k for k in keys if k.get("kid") == kid), None)
    if match is None:
        # Either the provider rotated its keys, or this token is not theirs.
        # One refetch tells the two apart.
        keys = (await _jwks(jwks_uri, force=True)).get("keys") or []
        match = next((k for k in keys if k.get("kid") == kid), None)
    if match is None and len(keys) == 1 and not kid:
        # A provider with exactly one key and no `kid` is unusual but legal.
        match = keys[0]
    if match is None:
        raise TokenError("The ID token was signed with a key this provider "
                         "does not publish.")

    _check_signature(header.get("alg") or match.get("alg") or "RS256",
                     _public_key(match), signed, signature)

    if payload.get("iss") != issuer:
        raise TokenError("That token was issued by %s, not by the provider "
                         "this studio is configured for."
                         % (payload.get("iss") or "nobody"))

    aud = payload.get("aud")
    aud = [aud] if isinstance(aud, str) else list(aud or [])
    if audience not in aud:
        raise TokenError("That token was issued for a different application.")
    # `azp` matters only when the token was minted for several applications:
    # it names the one that asked, and if that is not us the token is being
    # relayed to us by somebody else.
    if len(aud) > 1 and payload.get("azp") not in (None, audience):
        raise TokenError("That token was requested by a different application.")

    now = time.time()
    if float(payload.get("exp") or 0) < now - LEEWAY_S:
        raise TokenError("That sign-in took too long and the token expired. "
                         "Try again.")
    if float(payload.get("iat") or 0) > now + LEEWAY_S:
        raise TokenError("That token is dated in the future. Check the clock "
                         "on this machine.")
    if nonce is not None and payload.get("nonce") != nonce:
        raise TokenError("That sign-in did not match the one this browser "
                         "started. Try again from the sign-in page.")
    return payload


def unverified_claims(token: str) -> dict[str, Any]:
    """The payload without checking anything -- for showing, never deciding.

    Used only to display what a provider sent while an administrator is
    setting it up. Nothing that grants access may call this.
    """
    try:
        return _segments(token)[1]
    except TokenError:
        return {}
