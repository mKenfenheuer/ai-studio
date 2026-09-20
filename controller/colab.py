"""Google Colab as a runner: the notebook, and the code it downloads.

Google lends out a T4 for a few hours at a time, and a T4 fine-tunes a 7B in
4-bit perfectly well. What makes that usable here is the direction the runner
already dials. Colab has no inbound address and never will -- no port to
forward, no IP to reach -- and the agent has needed neither since the day it
was written. So "borrow Google's GPU" reduces to getting the runner's source
onto the VM and starting it, which is all this notebook does.

Two decisions worth stating, because both could have gone the other way:

**The runner code comes from this controller, not from PyPI or a git URL.** A
studio is frequently a private checkout on somebody's laptop, and a runner one
version behind its controller is a protocol mismatch nobody enjoys debugging.
The bundle is the `runner` and `common` packages exactly as this controller is
running them, so the two cannot drift.

**The join token is not written into the notebook.** An `.ipynb` is a file
people forward, commit and paste into issues, and the token is a password: it
attaches a machine to the studio and reads the work on it. So the notebook
asks for it instead -- out of Colab's own secret store when it is there -- and
never prints it back.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

# The repository root, whichever way this is installed. The controller's own
# Docker image carries `runner/` for exactly this reason: it installs none of
# the runner's dependencies (no torch, no CUDA), it only serves the source.
SOURCE_ROOT = Path(__file__).resolve().parent.parent

# What a runner needs to run: its own package, and the formatting code it
# shares with the controller so a conversation is rendered the same way in
# both places. Nothing else -- the controller's own code is not part of this.
BUNDLE_PARTS = ("runner", "common")

# Not source, and not optional. The agent reads its own version out of
# pyproject.toml and reports it in every capability probe; the Machines page
# reads that back and says so when a machine is behind. Leave the file out of
# the bundle and the version reads "unknown", which that page renders as "at
# least several releases behind" -- a loud, permanent and entirely false
# warning on a runner that is by construction the same version as the
# controller that handed it its code.
BUNDLE_FILES = ("pyproject.toml",)


def source_available() -> bool:
    """Whether this controller can hand out runner code at all.

    False on a deployment whose image predates this feature. Worth answering
    before the UI offers a download that would 404 in Colab ten minutes later.
    """
    return (all((SOURCE_ROOT / part / "__init__.py").exists()
                for part in BUNDLE_PARTS)
            and all((SOURCE_ROOT / f).exists() for f in BUNDLE_FILES))


def bundle(version: str) -> bytes:
    """The runner's source as a zip, laid out to run with `python -m runner`.

    Deliberately not a wheel and not a pip install. The Docker runner images
    do exactly this -- copy the two packages in, run the module -- and it means
    Colab installs third-party libraries only, so nothing can quietly replace
    the torch build Google matched to the GPU in that VM.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        count = 0
        for part in BUNDLE_PARTS:
            for f in sorted((SOURCE_ROOT / part).rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f.relative_to(SOURCE_ROOT).as_posix())
                count += 1
        for name in BUNDLE_FILES:
            z.write(SOURCE_ROOT / name, name)
            count += 1
        z.writestr("bundle.json", json.dumps({
            "version": version,
            "files": count,
            "built": int(time.time()),
        }, indent=2))
    return buf.getvalue()


# ===========================================================================
# The notebook
# ===========================================================================

INTRO = """\
# Lend this Colab's GPU to AI Studio

Running the three cells below turns this session into a **machine in your
studio**: it downloads the runner from your controller, starts it, and holds
the connection open for as long as Colab lets the session live. Your datasets,
your runs and your finished models stay on your controller — only the training
happens here.

Open your studio's **Machines** page while this runs. This session appears
there within a few seconds of the last cell starting, with whatever card
Google handed out today.

---

### Three things to know first

**1 · Turn the GPU on.** *Runtime → Change runtime type → T4 GPU*. Without it
this attaches a CPU machine, which does train — about fifty times slower.

**2 · Colab has to be able to reach your controller.** The runner dials out,
so Colab needs no open port — but `localhost` here means *this container*, not
your computer. If your studio runs on your own network, put a tunnel in front
of it first. On the machine running the controller:

    cloudflared tunnel --url http://localhost:8420

That prints an `https://….trycloudflare.com` address to paste below. Anyone
holding that address and the join token can attach a machine to your studio
and read the work on it, so treat the pair as a password, and stop the tunnel
when you are done.

**3 · The session will end.** Colab reclaims runtimes after a few hours, and
sooner if you close the tab. Nothing on the controller is lost: a run that was
training here goes back on the queue. Its checkpoint was on this VM's disk,
which Colab wipes, so the studio waits ten minutes for this session to come
back and then starts that run again on whichever machine is free.
"""

SETUP = '''\
#@title 1 · Point this Colab at your studio

#@markdown The address your controller answers on, as seen from the internet.
CONTROLLER_URL = "__CONTROLLER_URL__"  #@param {type:"string"}

#@markdown What this machine is called on the Machines page. Two Colab sessions
#@markdown on one studio need two different names — the name is what gives this
#@markdown machine the same identity each time it reconnects, rather than
#@markdown leaving a row of ghosts behind it.
RUNNER_NAME = "Colab"  #@param {type:"string"}

import getpass, hashlib, json, os, re, subprocess
import urllib.error, urllib.parse, urllib.request

CONTROLLER_URL = CONTROLLER_URL.strip().rstrip("/")
SRC  = "/content/ai-studio"        # the runner's own code
DATA = "/content/ai-studio-data"   # model cache, checkpoints, HF downloads

STUDIO_READY = False


def stop(problem, *fix):
    """Refuse here, with the fix, rather than failing two cells later."""
    print("\\n\\u2717 " + problem)
    for line in fix:
        print(("  " + line) if line else "")
    raise SystemExit(1)


def studio(path, token=None, timeout=30):
    req = urllib.request.Request(CONTROLLER_URL + path)
    if token:
        req.add_header("X-Runner-Token", token)
    return urllib.request.urlopen(req, timeout=timeout)


# ---- the card ------------------------------------------------------------
# Asked of nvidia-smi rather than of torch. Importing torch here would open a
# CUDA context in the notebook's own process and hold a few hundred megabytes
# of a 15 GB card for the rest of the session — beside the runner, which is
# about to open one of its own and is the process that needs the memory.
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60)
    GPU = smi.stdout.strip().splitlines()[0] if smi.returncode == 0 else ""
except (OSError, subprocess.SubprocessError, IndexError):
    GPU = ""

if GPU:
    print("GPU        : " + GPU)
else:
    print("GPU        : none")
    print("             Runtime > Change runtime type > T4 GPU, then run this")
    print("             cell again. Attaching without one is only worth it for")
    print("             writing a dataset with a hosted model, which uses no GPU.")

# ---- the address ---------------------------------------------------------
u = urllib.parse.urlparse(CONTROLLER_URL)
if u.scheme not in ("http", "https") or not u.hostname:
    stop("'%s' is not an address I can use." % CONTROLLER_URL,
         "It should look like https://studio.example.com, or the",
         "https://….trycloudflare.com address a tunnel prints.")

host = u.hostname
if (host in ("localhost", "::1") or host.endswith(".local")
        or re.match(r"^(127|10|0)\\.", host)
        or re.match(r"^192\\.168\\.", host)
        or re.match(r"^169\\.254\\.", host)
        or re.match(r"^172\\.(1[6-9]|2[0-9]|3[01])\\.", host)):
    stop("%s is an address on your own network, and this notebook is not on it."
         % host,
         "Colab runs in a Google datacentre. It can only reach a controller",
         "that has an address on the internet.",
         "",
         "On the machine running the controller:",
         "",
         "    cloudflared tunnel --url http://localhost:8420",
         "",
         "then paste the https://….trycloudflare.com address it prints into",
         "CONTROLLER_URL above. Anyone holding that address and the join token",
         "can attach a machine to your studio, so stop the tunnel afterwards.")

try:
    with studio("/api/health") as r:
        json.loads(r.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    stop("%s answered %d, which is not an AI Studio controller."
         % (CONTROLLER_URL, e.code),
         "Check the address — a tunnel pointed at the wrong port does this.")
except Exception as e:
    stop("Could not reach %s from Colab (%s)."
         % (CONTROLLER_URL, e.__class__.__name__),
         "The controller has to be running, and reachable from the internet.",
         "If it sits on your own network, put a tunnel in front of it — see",
         "the note at the top of this notebook.")
print("Controller : %s, reachable" % CONTROLLER_URL)

# ---- the token -----------------------------------------------------------
# Deliberately not stored in this file: an .ipynb gets forwarded, committed and
# pasted into issues, and this token attaches machines to your studio.
TOKEN = (os.environ.get("AI_STUDIO_JOIN_TOKEN") or "").strip()
SOURCE = "this session's environment"
if not TOKEN:
    try:
        from google.colab import userdata
        TOKEN = (userdata.get("AI_STUDIO_JOIN_TOKEN") or "").strip()
        SOURCE = "Colab secrets"
    except Exception:
        TOKEN = ""
if not TOKEN:
    print("")
    print("Paste the join token from your studio's Machines page. It is not")
    print("echoed and not saved into this notebook. To skip this next time,")
    print("put it in Colab's secrets — the key icon in the left sidebar —")
    print("under the name AI_STUDIO_JOIN_TOKEN, and give this notebook access.")
    TOKEN = getpass.getpass("Join token: ").strip()
    SOURCE = "typed just now"
if not TOKEN:
    stop("No join token, so there is nothing to connect with.",
         "It is on the Machines page of your studio, beside these commands.")
print("Token      : %s" % SOURCE)

STUDIO_READY = True
print("")
print("Ready. Run the next cell.")
'''

INSTALL = '''\
#@title 2 · Fetch the runner and the libraries it needs
if not globals().get("STUDIO_READY"):
    raise SystemExit("Run the cell above first — it checks these settings.")

import io, json, os, shutil, subprocess, sys, urllib.error, zipfile

# Everything the runner imports, except torch. Colab's torch is built against
# the driver in this VM and is the one thing here that must not be replaced:
# pip will happily install a wheel for a different CUDA and leave you training
# on the CPU, or with no working GPU at all.
print("Installing libraries (a minute or two)...")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers>=4.44", "peft>=0.12", "accelerate>=0.34",
                "datasets>=2.20", "safetensors>=0.4", "websockets>=12",
                "httpx>=0.27", "bitsandbytes>=0.43"], check=True)

# The runner's own code, from your controller rather than from a package
# index, so it is exactly the version your studio speaks.
print("Downloading the runner from your studio...")
try:
    with studio("/api/runner/bundle.zip", TOKEN, timeout=180) as r:
        blob = r.read()
except urllib.error.HTTPError as e:
    if e.code in (401, 403):
        stop("Your studio did not accept that join token.",
             "Copy it again from the Machines page and re-run the cell above.")
    elif e.code == 404:
        stop("This studio does not hand out runner code.",
             "It is running a version from before this notebook existed.",
             "Update the controller, then download the notebook again.")
    raise

shutil.rmtree(SRC, ignore_errors=True)
os.makedirs(SRC, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    z.extractall(SRC)
with open(os.path.join(SRC, "bundle.json")) as fh:
    info = json.load(fh)
os.makedirs(DATA, exist_ok=True)

print("")
print("Runner     : version %s, %d files, unpacked into %s"
      % (info["version"], info["files"], SRC))
print("Ready. Run the next cell to connect.")
'''

RUN = '''\
#@title 3 · Connect — leave this cell running
if not globals().get("STUDIO_READY"):
    raise SystemExit("Run the cells above first.")

import hashlib, os, signal, subprocess, sys, threading, time

# Same studio and same name means the same machine, every time. Without this
# the runner invents an identity on first start, Colab wipes it with the VM,
# and the Machines page fills up with the ghosts of yesterday's sessions.
idfile = os.path.join(DATA, "runner-id")
if not os.path.exists(idfile):
    seed = hashlib.sha256(("%s|%s" % (CONTROLLER_URL, RUNNER_NAME)).encode())
    with open(idfile, "w") as fh:
        fh.write("run_colab" + seed.hexdigest()[:10])

env = dict(os.environ)
env.update({
    "AI_STUDIO_CONTROLLER": CONTROLLER_URL,
    "AI_STUDIO_JOIN_TOKEN": TOKEN,
    "AI_STUDIO_RUNNER_NAME": RUNNER_NAME,
    "AI_STUDIO_RUNNER_STATE": idfile,
    # All under /content, which is the VM's real disk. Models are gigabytes
    # and the default locations are not sized for them.
    "AI_STUDIO_CHECKPOINTS": os.path.join(DATA, "checkpoints"),
    "AI_STUDIO_MODEL_CACHE": os.path.join(DATA, "models"),
    "HF_HOME": os.path.join(DATA, "hf"),
    "PYTHONUNBUFFERED": "1",
})

print("Starting the runner. It benchmarks the card first, which takes a minute")
print("and is what lets your studio size a run for this machine honestly.")
print("Leave this cell running — stopping it disconnects the machine.")
print("")

proc = subprocess.Popen([sys.executable, "-m", "runner"], cwd=SRC, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)

# A connected runner with nothing to do says nothing at all, which after an
# hour is indistinguishable from a dead cell. One line every fifteen minutes.
started = time.time()


def _still_here():
    while proc.poll() is None:
        time.sleep(900)
        if proc.poll() is None:
            mins = int((time.time() - started) // 60)
            print("  · still connected — %dh %02dm" % (mins // 60, mins % 60),
                  flush=True)


threading.Thread(target=_still_here, daemon=True).start()

try:
    for line in proc.stdout:
        print(line, end="", flush=True)
except KeyboardInterrupt:
    print("\\nDisconnecting...")
    try:
        # The agent catches this and shuts the socket down politely. Not every
        # platform can send it -- Windows cannot, and somebody will run this
        # notebook locally -- so fall back to ending it the blunt way.
        proc.send_signal(signal.SIGINT)
    except (ValueError, OSError):
        proc.terminate()
finally:
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()

print("")
print("Disconnected. Run this cell again to rejoin as the same machine.")
'''

OUTRO = """\
---

### While it runs

Your studio's **Machines** page now lists this session: the card, the dtype it
measured as fastest, whether 4-bit works here (on a T4 it does), and the
largest model it thinks this machine can fine-tune. Queue a run and it lands
on this GPU.

### When it stops

Run the last cell again and it rejoins under the same name. If Colab took the
whole runtime back, run all three cells — the code is downloaded again and
there is nothing to clean up.

### When something is wrong

| What you see | What it means |
| --- | --- |
| `Could not reach …` | Colab cannot see your controller. It needs an address on the internet — see the tunnel note at the top. |
| `did not accept that join token` | The token has changed, or was copied short. Take it again from the Machines page. |
| `GPU : none` | *Runtime → Change runtime type → T4 GPU*, then run all three cells again. |
| Connected, but never given a run | The studio sends one job to one machine at a time. While another machine is training, this one waits. |
| A run dies out of memory | A T4 holds 15 GB. Choose 4-bit, or a smaller model — the studio's size picker costs both against this card. |

### What it costs

Nothing beyond the Colab session itself, and what gets trained is yours: the
model is uploaded to your controller when the run finishes, and stays there
long after this session is gone.
"""


def notebook(controller_url: str) -> dict:
    """The .ipynb, with this studio's address already filled in."""
    return {
        "cells": [
            _cell("markdown", INTRO),
            _cell("code", SETUP.replace("__CONTROLLER_URL__", controller_url)),
            _cell("code", INSTALL),
            _cell("code", RUN),
            _cell("markdown", OUTRO),
        ],
        "metadata": {
            # Colab reads both of these when the notebook is opened and offers
            # the matching runtime, which heads off the single most common way
            # to get this wrong: attaching a CPU machine by accident.
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4",
                      "name": "ai-studio-runner.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _cell(kind: str, src: str) -> dict:
    """One notebook cell. nbformat stores the source as lines, each keeping
    its newline except the last -- not as one string."""
    cell = {"cell_type": kind, "metadata": {},
            "source": src.rstrip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell
