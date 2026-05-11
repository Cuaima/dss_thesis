# Annotation Manual — Depression Connect Support Detection
*For second rater use*

---

## What you are doing

You will receive a CSV file (`annotation_sample_with_context.csv`) containing 800 replies from an anonymous Dutch depression support forum. Each row is one reply. Your task is to read each reply and fill in two columns: `label` and `support_type`.

You do not need to read the `initial_post_text` column for every message, but it is there as context if a reply is unclear without knowing what it is responding to. **For edge cases, always read the initial post first.**

Note: some text in the messages will appear as anonymization placeholders such as `entity_person_1` or `entity_location_2`. These are not errors — they represent names and places that were removed to protect user privacy. Read through them as if they were the original word.

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
| `other` | The reply is supportive but does not clearly fit either category above (e.g. asking caring questions, expressing that the person belongs in the community, or complimenting their strength) |

> A reply can sometimes feel like both informational and emotional. In that case, pick whichever feels **most dominant**. There is no wrong answer here as long as you can justify your choice.

---

## Edge cases

### Replies that share personal experience

A reply that shares the replier's own experience or story can be either SS or NSS depending on **who the reply is really about**.

- If the personal experience is shared to **validate, comfort, or advise** the original poster → label `SS`
- If the reply is primarily **about the replier themselves** with little connection back to the original poster → label `NSS`

**Ask yourself: is the original poster the subject of this reply, or is the replier?**
- Original poster is the subject → `SS`
- Replier is the subject → `NSS`

For `SS` replies sharing personal experience, classify by content:
- Experience contains practical advice or information → `SS - informational`
- Experience is primarily emotional or about feelings → `SS - emotional`
- Experience is general solidarity without a clear dominant type → `SS - other`

---

### Replies to experience-soliciting posts

When the initial post explicitly **asks others to share their experiences** (e.g. *"wat is jullie ervaring met X?"* / "what is everyone's experience with X?"), replies that share personal experience are `SS` by default — they are directly fulfilling the support request made by the original poster.

Classify by content:
- Practical information or advice about the topic → `SS - informational`
- Emotional experience or feelings about the topic → `SS - emotional`
- General experience sharing without a clear dominant type → `SS - other`

The exception is a reply that **ignores the question entirely** and talks about something unrelated → `NSS`

---

### Replies that only ask questions

A reply consisting entirely of questions can be SS or NSS depending on tone.

- If the questions express **genuine interest in the person's wellbeing** and would make the original poster feel seen and cared for → label `SS`, `other`
- If the questions are purely **clarifying or informational** with no emotional warmth, or feel interrogative rather than caring → label `NSS`

**Ask yourself: would the original poster feel *cared for* or *interrogated* after reading this reply?**
- Cared for → `SS - other`
- Interrogated → `NSS`

---

## Quick reference card

| Situation | `label` | `support_type` |
|-----------|---------|----------------|
| Reply expresses empathy or care | `SS` | `emotional` |
| Reply gives advice or practical tips | `SS` | `informational` |
| Reply shares personal experience as validation or advice | `SS` | `emotional` or `informational` |
| Reply shares personal experience as general solidarity | `SS` | `other` |
| Reply asks caring questions about the OP's wellbeing | `SS` | `other` |
| Reply to an experience-soliciting post shares experience | `SS` | depends on content |
| Reply is supportive but fits none of the above | `SS` | `other` |
| Reply is very short with no support ("ok", "thanks") | `NSS` | `N/A` |
| Reply is primarily about the replier with no connection to OP | `NSS` | `N/A` |
| Reply is off-topic or neutral | `NSS` | `N/A` |
| Reply is only a quoted block | `NSS` | `N/A` |
| Reply asks purely clarifying questions with no warmth | `NSS` | `N/A` |

---

## Practical instructions

1. Open the CSV in Excel or Google Sheets
2. Work through rows one at a time — do not skip around
3. For each row, **check the `initial_post_text` column** if the reply is unclear on its own
4. Fill in `label` first, then `support_type`
5. Do not change any other columns
6. Save the file when done and return it with your name in the filename
   - e.g. `annotation_sample_rater2.csv`
7. If you are unsure about more than a handful of cases, contact me **before** finishing so we can discuss and align

> **Important:** do not discuss specific cases with me until both of us have finished annotating independently. This is necessary for the reliability of the study.

---

## Time estimate

Approximately **3–4 hours** for 800 messages, depending on your reading speed.
You do not need to finish in one sitting.

---

## Questions?

Contact me at [your email address].

Thank you for your help — your contribution directly supports the validity of this thesis.
