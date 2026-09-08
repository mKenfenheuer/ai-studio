/**
 * Which machine should do a piece of work.
 *
 * Every page that offers a choice of machine used to take the first one the
 * API happened to return, which is ordered by when each machine first joined.
 * On a studio whose CPU-only controller joined first, that meant the default
 * everywhere was the machine that cannot train anything -- and a machine with
 * no card cannot say whether a model fits, and offers no 4-bit, so the
 * settings that depend on those quietly went missing rather than saying why.
 *
 * One order, used by all of them: what can do the most, first.
 */

/** Machines sorted the way somebody would choose them: cards before no card,
 *  more memory before less, idle before busy at equal capability -- work that
 *  can start now beats a slightly larger card with a queue in front of it. */
export function byCapability(runners) {
  const rank = (r) => {
    const c = r.capabilities || {};
    return [c.backend && c.backend !== "cpu" ? 1 : 0,
            c.vram_gb || 0,
            r.status === "busy" ? 0 : 1];
  };
  return [...(runners || [])].sort((a, b) => {
    const [ga, va, fa] = rank(a);
    const [gb, vb, fb] = rank(b);
    return gb - ga || vb - va || fb - fa;
  });
}

/** The machines worth offering, best first. */
export const usableMachines = (runners) =>
  byCapability((runners || []).filter((r) => r.status !== "offline"));

/** The one to select when nobody has said. */
export const bestMachine = (runners) => usableMachines(runners)[0] || null;

/** What a machine cannot do, as badges to put on its tile. Said on the face
 *  of it, because "no 4-bit" explains a greyed-out setting two steps later. */
export function machineLimits(runner) {
  const c = runner?.capabilities || {};
  const out = [];
  if (!c.vram_gb) {
    out.push({ label: "no GPU",
               why: "No graphics card: fine for trying the app out, far too "
                  + "slow for real training" });
  } else if (!(c.quantization || {})["4bit"]) {
    out.push({ label: "no 4-bit",
               why: "Without 4-bit a model has to fit in 16-bit — about four "
                  + "times the memory" });
  }
  return out;
}
