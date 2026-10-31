# Data license

The expert annotations under `data/conversation/annotations/` and the result tables under `results/` are released
under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
license. The code is under the [MIT license](LICENSE).

## What is released

- `data/conversation/annotations/expert1.docx`, `expert2.docx`: the two experts' filled-in annotation sheets, one
  per annotator, over the 34 conversation phases that the screening step flagged. Each phase shows the cleaned
  transcript, a checkbox for whether loneliness is expressed, and the annotator's quoted evidence.
- `data/conversation/annotations/annotations.csv`: the same 34 phases as one table with both labels, both
  explanations and both span sets as columns. Nothing is adjudicated.
- `data/conversation/annotations/annotator_agreement.csv`: percent agreement, Cohen's kappa, Gwet's AC1 and the
  word-level span overlap between the two annotators.

## What is not released

The full conversation corpora (383 sessions of 16 participants) contain private information and are not released.
The 34 released phases are the ones two experts annotated, and they have been de-identified: personal names
(including pets and the chatbot) are replaced by `[NAME]` and places by `[PLACE]`; participants are identified only by
study pseudonyms (`US-xxxx`) and sessions by opaque ids. Participants gave informed consent under an IRB-approved
protocol for this use.

## LonXplain

The LonXplain corpus (Garg et al., 2023) is not redistributed here. Obtain it from its authors and place it at
`data/lonxplain/raw/Lonesomenessdataset.csv`; `data/README.md` describes the file the pipeline expects.

## Attribution

When you use the annotations, cite the paper (see the Citation section of the README). The labels are two experts'
independent readings of loneliness *expression* in a conversation phase, not a clinical assessment of the participant.
