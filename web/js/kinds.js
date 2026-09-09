/**
 * What each kind of run is, in one place.
 *
 * Seven facts about a run — what to call it, what it is doing while it runs,
 * what its progress numbers count, what to say it is working on, whether it
 * leaves a model behind, whether two of them can be ranked against each other,
 * and which page shape it wants — were spread across five files and four
 * near-identical switch statements: `RUNNING_LABEL` and `UNIT` in util.js,
 * `subject()` in dashboard.js, `what()` in jobs.js, and `STAGES`/`KIND_STAGES`
 * plus `OWN_PAGE` in job.js. Adding a kind meant finding all of them, and the
 * ones that were missed failed quietly: a run writing rows announced itself as
 * training, and a scoring run counted its answers in steps.
 *
 * So: one table. A new kind is an entry here and a page shape, and everything
 * that reads a run picks it up.
 */
import { esc } from "./util.js";

/** Families that can be ranked against one another. A from-scratch model's
 *  held-out loss is measured against its own small vocabulary and a
 *  fine-tune's against a borrowed 150k one; putting both in one table is a
 *  league table of unrelated numbers. */
export const FAMILY = { TUNED: "Fine-tunes", SCRATCH: "From scratch",
                        CLASSIFIER: "Classifiers" };

export const KINDS = {
  finetune_llm: {
    label: "Fine-tune",
    icon: "🎯",
    running: "Training",
    unit: null,                     // steps
    page: "training",
    leavesModel: true,
    family: FAMILY.TUNED,
    subject: (c) => c.base_model_label || c.base_model || "a base model",
  },
  pretrain_llm: {
    label: "From scratch",
    icon: "🌱",
    running: "Training",
    unit: null,
    page: "training",
    leavesModel: true,
    family: FAMILY.SCRATCH,
    subject: (c) => "from scratch · " + (c.dataset_label || c.dataset || "text"),
  },
  finetune_vision_cls: {
    label: "Classifier",
    icon: "🖼",
    running: "Training",
    unit: null,
    page: "training",
    leavesModel: true,
    family: FAMILY.CLASSIFIER,
    subject: (c) => (c.base_model_label || c.base_model || "a vision model")
      + " on " + (c.dataset_label || "pictures"),
    stages: {
      loading_dataset: "Fetching the pictures…",
      loading_model: "Downloading and loading the model…",
      training: "Training",
      evaluating: "Checking it on the held-out pictures…",
    },
  },
  generate_dataset: {
    label: "Dataset",
    icon: "✦",
    running: "Writing rows",
    unit: "rows",
    page: "writing",
    leavesModel: false,
    family: null,
    subject: (c) => "writing rows with "
      + (c.model?.label || c.model?.model || c.model?.base_model || "a model"),
  },
  upload: {
    label: "Publish",
    icon: "☁",
    running: "Uploading",
    unit: "MB",
    page: "upload",
    leavesModel: false,
    family: null,
    subject: (c) => "→ " + (c.repo_id || "Hugging Face"),
  },
  evaluate: {
    label: "Scoring",
    icon: "◎",
    running: "Scoring",
    unit: "answers",
    page: "eval",
    leavesModel: false,
    family: null,
    // Which models, not just which set. Three scorings of the same set look
    // identical otherwise -- same generated name, same subject line -- and
    // the thing that differs between them is exactly what was scored.
    subject: (c) => {
      const who = (c.models || []).map((m) => m.name).filter(Boolean);
      const against = who.length > 2
        ? `${who.slice(0, 2).join(", ")} and ${who.length - 2} more`
        : who.join(" vs ");
      return "scoring " + (against || "models")
             + " against " + (c.eval_name || "a prompt set");
    },
  },
  export_gguf: {
    label: "GGUF export",
    icon: "⬓",
    running: "Converting",
    unit: "stages",
    page: "upload",
    leavesModel: false,
    family: null,
    subject: (c) => "converting " + (c.source_run_name || "a model")
                    + " to " + (c.quantize || "GGUF"),
  },
  // Kept because runs of this kind exist in databases that predate the merge
  // moving into the fine-tune that produces it. Nothing creates one now.
  merge_adapter: {
    label: "Merge",
    icon: "⧉",
    running: "Merging",
    unit: "stages",
    page: "merge",
    leavesModel: true,
    family: FAMILY.TUNED,
    legacy: true,
    subject: (c) => "folding into " + (c.base_model_label || c.base_model || "its base"),
    stages: {
      loading_model: "Loading the adapter and the weights it was trained on…",
      saving: "Writing the merged model…",
    },
  },
};

const FALLBACK = {
  label: "Run", icon: "•", running: "Working", unit: null, page: "training",
  leavesModel: false, family: null, subject: () => "",
};

export const kindOf = (job) => KINDS[job?.kind] || FALLBACK;

/** What this run is working on, in one line. */
export const subjectOf = (job) => {
  try { return kindOf(job).subject(job?.config || {}) || ""; }
  catch { return ""; }
};

/** The stage wording for a run, with this kind's overrides applied. */
export const stagesFor = (job) => ({ ...STAGES, ...(kindOf(job).stages || {}) });

/** Kinds that leave something to talk to, score and publish. */
export const modelKinds = () =>
  Object.keys(KINDS).filter((k) => KINDS[k].leavesModel);

export const kindBadge = (job) => {
  const k = kindOf(job);
  return `<span class="badge" title="${esc(k.label)}">${k.icon} ${esc(k.label)}</span>`;
};

/** What a run is doing right now, per stage. The words are the same for every
 *  kind unless that kind says otherwise: the same stage means something
 *  different depending on what is running. */
export const STAGES = {
  evaluating: "Putting the prompts to each model…",
  loading_model: "Downloading and loading the model…",
  // Writing a dataset is not training, and a hosted writer is not downloaded.
  // These runs used to borrow the training vocabulary and report that they
  // were loading a model and then training, while they were opening an HTTPS
  // connection and then writing rows.
  connecting: "Reaching the model that will write it…",
  writing: "Writing rows",
  collecting: "Collecting the files to send…",
  uploading: "Uploading to Hugging Face",
  loading_dataset: "Downloading and preparing your data…",
  training_tokenizer: "Building a vocabulary from your text…",
  tokenizing: "Reading and tokenizing the text…",
  building_model: "Creating the model from random weights…",
  training: "Training",
  saving: "Saving the result…",
  converting: "Reading the weights and writing one file…",
  quantizing: "Shrinking it to the size you asked for…",
};


/** What a run says it should be judged by, and which way is up.
 *
 *  `{name, label, value, lower}` or null. A run finished before it said
 *  carries only `best_val_loss`, read as a held-out loss, lower better --
 *  which is what it always was. Every page that ranks runs reads this and
 *  nothing else, so a classifier's accuracy and a language model's loss
 *  sort correctly side by side without the page knowing which is which. */
export function primaryMetric(job) {
  const s = job?.summary || {};
  const m = s.primary_metric;
  if (m && m.value != null) {
    return { name: m.name || "metric", label: m.label || "Metric",
             value: +m.value, lower: m.lower_better !== false };
  }
  if (s.best_val_loss != null) {
    return { name: "held_out_loss", label: "Held-out loss",
             value: +s.best_val_loss, lower: true };
  }
  return null;
}

/** The better of two values under a metric's polarity. */
export const betterOf = (m, a, b) => (m?.lower === false ? Math.max(a, b) : Math.min(a, b));
