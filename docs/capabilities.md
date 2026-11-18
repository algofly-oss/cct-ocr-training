# CCT-S capabilities and reference comparisons

This repository focuses on compact plate OCR. The table also places its measured scope beside the WATE and Florence references; it does not turn those models into CCT variants.

## Scenario evidence

“Not measured” means this reference collection does not establish the capability. It is not a zero score. Architecture support, qualitative experience and measured accuracy are different kinds of evidence.

| Scenario | CCT-S plate model | WATERec references | Florence reference |
|---|---|---|---|
| Upright plate crops | Historical engine literal replay:2,315/2,693 validation and 644/984 separate unseen plates. Timing uses a different256-image cohort and is not a clean-new-training qualification. | ALPR: 2,475/2,693 exact; separate reviewed unseen cohort 880/984; extracted fields 332/443. These denominators must not be combined. | Historical optimized engine:2,381/2,693 validation and 857/984 separate unseen plates; different training histories. |
| Two-line plates | Whole crop is resized to 64×128; no separately qualified two-line subset. | Native 2D patches retain spatial arrangement, but no isolated two-line accuracy is reported here. | Layout capability should be evaluated on the actual two-line cohort; no figure supplied here. |
| Native 90°/180°/270° reading | Not measured. Fixed input shape does not provide rotation invariance. | An older native FP32 checkpoint scored 247/788, 159/788, 251/788 on clockwise rotations, without correction. This is a separate diagnostic, not the current final-engine result. | No matched current rotation qualification claimed. |
| Rotation with correction | No qualified correction stage in this repository. | Optional experimental orientation pipeline: 706/788, 706/788, 707/788 at clockwise 90°/180°/270°. Upright reviewed cohort: 879/984 versus 880/984 without correction. These are pipeline results, not intrinsic reader invariance. | No correction result supplied here. |
| Unseen countries and fonts | Qualitative out-of-distribution weaknesses have been observed; no country-stratified score is supplied. | The reviewed unseen plate result is evidence for that cohort, not every country or script. | No universal country coverage claim. |
| Price: tiny raised fractions, currency, unequal glyph sizes | This 37-class plate model is not the price model. Its alphabet cannot represent decimal punctuation or spaces. | Historical R2048 price:153/206 literal exact. Detail preservation still matters; not a guarantee for every raised fraction. | Same historical price references:148/206 literal exact. This is not universal equivalence. |
| Extra text, clutter, few-shot adaptation | No demonstrated few-shot selective-reading capability. Training on plate crops does not establish price-field selection. | Current full-data scores are not 50-shot results. Selection among distractors remains task/data dependent. | User-observed useful adaptation from roughly 50 examples with clutter is qualitative evidence here, not a matched 90% benchmark or a guarantee. |
| Arbitrary source resolution | Images are accepted and resized to fixed 64×128. This is not preservation of arbitrary-resolution detail. | Aspect-preserving patch-4 preprocessing uses a configurable token budget. Actual dimensions/token counts depend on the image; changing budget requires fresh accuracy and resource checks. | This comparison does not qualify arbitrary-resolution export behavior. |
| Batch throughput and memory | Historical B1: 0.4474 ms, 2,235 images/s; B32: 2.5137 ms/batch, 12,730 images/s; both sampled 742 MiB. | Qualified static B1: plate 1.2558 ms / 694 MiB; price R2048 1.6838 ms / 716 MiB. Separate dynamic-batch implementations must be measured independently. | No directly comparable timing supplied in this table. |

All listed timing uses predecoded RGB, including preprocessing, transfer, inference and text decoding, with the relevant engine/runtime. CCT timing used 256 reference crops, five warmups and three passes, a batch-32 engine profile and batch-32 allocations. WATE plate and price timings have different cohorts and profiles. These are not architecture-only or hardware-independent comparisons. See the repository's reference metadata for evidence hashes and exact scope. Dummy-data runs establish wiring only.

## Choosing an experiment

For familiar, upright, tightly cropped alphanumeric plates with strict throughput constraints, CCT-S is a useful compact baseline. Its 15 parallel character slots avoid an autoregressive decode loop. Validate unfamiliar layouts and countries rather than assuming the speed result implies robustness.

For mixed glyph scales, currency and richer spatial layouts, the retained 2D WATE encoder plus autoregressive reader is the stronger measured price reference in this collection. Higher token budgets can retain smaller details but increase computation. PyTorch versus TensorRT is an execution choice, not the reason one model handles spatial layout better; exported precision, preprocessing and decoding must remain qualified.

Florence is a useful generative reference when selective reading and few-example adaptation matter. The user observation about clutter motivates testing, but does not substitute for a matched held-out evaluation. A larger generative model is not automatically the fastest deployment choice.

## Limits and model identities

- WATE R and RS are the same approximately 26.12M-parameter architecture with different pretrained initialization, not different size tiers. The price SPACE extension changes the classifier mapping and adds 768 parameters.
- FastTiny/PARSeq is a separate model family, not a smaller WATE variant. Unqualified custom 5.5M experiments are not public supported variants.
- The WATE export described here uses an ASCII character set and at most 25 decode steps including EOS. The default safe text length is 24 characters plus EOS; configuration accepting length 25 does not guarantee a terminating EOS within 25 steps. This is not universal Unicode OCR.
- CCT-S uses `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_`, with `_` as padding, and 15 slots. The generic upstream checkpoint's 10-slot/region head differs from the local 15-slot plate model.
- Orientation correction is optional experimental plate-pipeline evidence. No price-orientation qualification or automatic best-angle oracle is implied.
- Train/validation selection must precede final testing. Separate ALPR and price weights are allowed; passing one task does not qualify the other.

## Current corrected-price research is a separate pipeline

The public default is a37-class plate reader with 15 slots, OpenCV linear resizing and first-PAD termination. The historical price model has 39 classes and 977,736 learned parameters; the corrected-policy CCT64 research model has 41 classes and 977,962 learned parameters. The current research pipeline uses Pillow BICUBIC stretching and removes all PAD slots. No research result is inherited by the public decoder.

Current CCT64 scores 66/144 literal exact (45.83%) with 0.746 ms RGB /0.937 ms encoded and 542 MiB process GPU peak on RTX5090/B1. All 144 native strings/tokens were reproduced. CCT128 has 1,149,994 learned parameters and scores 65/144 (45.14%); its inference cost is unmeasured. Its 3,072 MiB is training memory. Different microbatch/RNG partition prevents an isolated resolution claim. These development-selected results use676 FIT and all 144 DEV images, not the historical 206 price set. The separate clean38-class ALPR checkpoint now scores953/1,167 (81.66%) and has own TensorRT costs0.846 ms RGB /1.032 ms encoded,542 MiB. It is distinct from the public37-class plate workflow; no timing or accuracy is inherited by that default.

Only literal complete-string exact accuracy is reported: symbols, spaces, case and order all matter. Historical ancestry and selection policies can differ; no old run is claimed to have used the current policy. See the [maintained comparison](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md) and [exact-only PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/reports/ocr_comparison_2026-11-17.pdf). Training remains image plus expected text only, without boxes or character positions.

## Current price rotation diagnostic

On the same corrected price DEV144, native clockwise 90°/180°/270° literal counts are WATE:26/22/66, compact WATE:20/16/29, measured custom 97 model:1/4/1, native 98 candidate:0/3/2, CCT64 price:0/0/0 and current price Florence:45/4/49. Upright counts are 106/102/97/98/66/105 respectively; all upright saved strings/tokens match. No correction or angle oracle was applied. This is not ALPR, final-test or TensorRT rotation evidence.

Quarter-turn preprocessing differs: WATE preserves area at 256×128, custom models shrink to 128×64 (one-quarter of upright area), CCT stretches 64×128 and Florence uses 384×384;180° retains each upright geometry. These whole-pipeline results do not isolate encoder invariance. Tiny-model intrinsic price rotation remains unmet. The98 checkpoint gains six exact cases and loses five versus97; its own upright deployment measures1.539 /1.768 ms,552 MiB. The97 checkpoint remains a distinct measured point, not a borrowed timing. See the [complete native rotation table](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md#native-quarter-turn-reading-current-price-dev144) and [exact-only PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/reports/ocr_comparison_2026-11-17.pdf).


An **optional learned orientation pipeline** (1,962,662 learned parameters, including a275,524-parameter head) gives **97/97/97/98 of144** atCW0/90/180/270, compared with the default98/0/3/2. It loses one previously correct upright output. Its own TensorRT pipeline reproduces the576 native views; all-angle RGB mean is **1.961 ms,560 MiB**, with upright mean1.920 ms. Encoded lossless-PNG mean2.734 ms is not a matched JPEG comparison. The head was selected on these same144 DEV images, not a blind rotation test; the default upright workflow is unchanged.

Variable-shape support is not scale-invariant accuracy: the same98 checkpoint gives **98,88,89,48,9 exact matches of144** at bounds128×256,96×192,160×320,192×384,256×512 respectively. These inference-only trials do not rule out higher-resolution training; no speed claim or default change follows.

Measured few-shot development curves for the synthetic-stage reader are **77/144 (53.47%) with50 real FIT images,83/144 (57.64%) with100, and87/144 (60.42%) with250**, versus97/144 (67.36%) for its676-image control. These nested single-seed runs each used13,520 task exposures and144 DEV labels to select among21 checkpoints. They also inherit public pretraining,384,000 natural-image exposures from20,000 unique crops, and32,000 synthetic exposures. They are not cold50-label adaptation, do not establish a simple user workflow, and have no own deployment timings.


Current PPmedium104/144 has two qualified operating points: mixed TensorRT **0.884 ms RGB /1.087 ms encoded,654 MiB** preserves canonical CTC output tokens and literal strings while two internal frame alignments differ; original numeric/frame failures remain recorded. The global FP32 control retains frames and passes its frozen numeric gate at **1.413 /1.688 ms,588 MiB**. Neither is a universal equivalence claim. A newer1.687M photometric-replay checkpoint scores **101/144 (70.14%)**, selected before further adaptation; its own B1 FP32 TensorRT/Graph4 deployment preserves all144 native strings/tokens/EOS at1.545 ms RGB /1.737 ms encoded,552 MiB. Similar runtime to98 is not a speed-win claim. All are single-seed, development-selected results on the same144 images, not final-test evidence.


## Current fixed-checkpoint evaluation

The [maintained exact-only comparison](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md#current-fixed-checkpoint-heldout-results) now separates current checkpoints from historical models. Current ALPR WATE/PP/CCT/custom score2,476/2,321/2,226/2,180 exact of2,693 and855/606/628/438 of984, with all native/TRT strings and tokens retained. Clean Florence224 scores2,002/2,693 and797/984 natively; its own deployment cost is unmeasured and it is not the supplied strong Florence model. These historically reused cohorts are not newly blind tests;984 labels are AI-reviewed, not human-certified.

Current price206 exact counts are163 WATE,163 PP,152 custom101,159 Florence and91 CCT. All9 external photos remain, with6 accepted literal references and3 null/unscorable: WATE/custom each match1 of6; others0. Bounds acrossall9 are1–4 or0–3, not known nine-image accuracy. Mixed PP changes one already-incorrect206 string; other applicable backends retain every native string. Costs remain separately measured DEV results. No labels, weights, precision or input policies are changed using final results. See the [exact-only PDF](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/reports/ocr_comparison_2026-11-17.pdf).


New PP ALPR48/128 checkpoints have independent empirical DEV operating points:1001/1167 at0.760 ms RGB /0.939 ms encoded,650 MiB;1027/1167 at0.964 /1.094 ms,654 MiB. Both use15.580M parameters and batch1; spatial inputs are48×320 and128×320. Exact DEV counts equal native, but two strings change per model; strict output/numeric gates remain failed. The48 point gains one correct case and loses one;128 changes two already-incorrect strings. Separate strict-FP32 controls are slower. **Their own heldout counts now differ by backend**: PP48 native/mixed gives2341/2339 of2693 and624/626 of984; PP128 gives2367/2368 and533/534. PP128 improves DEV/final but worsens reviewed984 versus48, so no blanket recommendation follows. Mixed strict-output failures remain; none of the older983-checkpoint results or costs is inherited. See the [full comparison and qualification scopes](https://github.com/algofly-oss/waterec-ocr-training/blob/main/docs/model_comparison.md).


A separate PLM/control study remains pending and has no completed result in this report. General-purpose few-shot transfer/usability and the approximately1M-parameter WATE-like accuracy/speed goal remain unproven. Source archives contain code, not measured trained weights.


Completed studies now add a negative matched natural-CTC result (custom control/CTC:ALPR933/927,price98/92) and all three official-generic PP few-shot outcomes:66/11/90 selected of144 at50/100/250 labels, including PP100 epoch0 selection/terminal0. Custom few-shot values77/83/87 inherit different pretraining; N also uses144 DEV annotations and is not N total. No causal family superiority or unique failure cause is established. Own control933 FP32 TRT costs1.833/2.063 ms,552 MiB; clean Florence224 native BF16 costs34.966/34.947 ms,1696 MiB, on full1167 DEV. Do not inherit older929 or historical Florence costs. A separate PLM/control study remains pending, without a quality claim. See the maintained comparison for exact scope and limitations.
