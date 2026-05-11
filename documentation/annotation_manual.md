# Annotation Manual — Depression Connect Support Detection
*For second rater use*

---

## What you are doing

You will receive a CSV file (`annotation_sample_with_context.csv`) containing 800 replies from an anonymous Dutch depression support forum. Each row is one reply. Your task is to read each reply and fill in two columns: `label` and `support_type`.

You do not need to read the `initial_post_text` column for every message, but it is there as context if a reply is unclear without knowing what it is responding to.

---

## Column: `label`

This is the most important column. For each reply, decide whether it is socially supportive or not, and write one of these two values:

| Value | Meaning |
|-------|---------|
| `SS` | **Social Support** — the reply offers support, encouragement, empathy, advice, or acknowledgement to the person who posted. It makes the reader feel heard, less alone, or better equipped to cope. |
| `NSS` | **Not Social Support** — the reply does not convey support. This includes purely informational messages with no emotional warmth, off-topic replies, spam, very short acknowledgements ("ok", "thanks"), or messages that are neutral or negative in tone. |

### Examples of SS
- *"Ik herken dit zo goed, je bent niet alleen"* (I recognize this so well, you are not alone)
- *"Misschien kun je proberen om elke dag even naar buiten te gaan, dat hielp mij ook"* (Maybe you could try going outside for a bit each day, that helped me too)
- *"Wat moeilijk voor je, ik denk aan je"* (How hard for you, I'm thinking of you)

### Examples of NSS
- *"Ok"*
- *"Wanneer is de volgende bijeenkomst?"* (When is the next meeting?)
- A reply that is entirely a quoted block with no added text

### When in doubt
Ask yourself: *if I had posted this and received this reply, would I feel supported?*
- If **yes** → label `SS`
- If **no or unsure** → label `NSS`

---

## Column: `support_type`

Only fill this in **if you labeled the message `SS`**. If you labeled it `NSS`, write `N/A`.

For `SS` messages, decide which type of support is primarily being offered:

| Value | Meaning |
|-------|---------|
| `informational` | The reply gives advice, information, suggestions, or practical guidance |
| `emotional` | The reply expresses empathy, care, understanding, or emotional solidarity |
| `other` | The reply is supportive but does not clearly fit either category above (e.g. esteem support like complimenting someone's strength, or simply expressing that they belong in the community) |

> A reply can sometimes feel like both informational and emotional. In that case, pick whichever feels **most dominant**. There is no wrong answer here as long as you can justify your choice.

---

## Practical instructions

1. Open the CSV in Excel or Google Sheets
2. Work through rows one at a time
3. Fill in `label` first, then `support_type`
4. Do not change any other columns
5. Save the file when done and return it with the filename `annotation_sample_rater2.csv`
6. If you are unsure about more than a handful of cases, contact me **before** finishing so we can discuss and align

> **Important:** do not discuss specific cases with me until both of us have finished annotating independently. This is necessary for the reliability of the study.

---

## Quick reference card

| Situation | `label` | `support_type` |
|-----------|---------|----------------|
| Reply expresses empathy or care | `SS` | `emotional` |
| Reply gives advice or practical tips | `SS` | `informational` |
| Reply is supportive but neither of the above | `SS` | `other` |
| Reply is very short with no support ("ok", "thanks") | `NSS` | `N/A` |
| Reply is off-topic or neutral | `NSS` | `N/A` |
| Reply is only a quoted block | `NSS` | `N/A` |

---

## Time estimate

Approximately **3–4 hours** for 800 messages, depending on your reading speed.
You do not need to finish in one sitting.

---

Thank you for your help!
