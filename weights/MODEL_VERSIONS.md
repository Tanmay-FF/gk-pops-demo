# Cart classifier versions

The engine picks a version with `CLASSIFIER_BACKEND` in `engine/config.py`.
Both versions stay in this folder, so switching back is one line.

| | v1 (legacy) | v2 (multihead) |
|---|---|---|
| `CLASSIFIER_BACKEND` | `"legacy"` | `"multihead"` (current default since 2026-11-12) |
| Files | `cart_quality/weights/best.pt` + `cart_quality/calibration.json`; `fill_and_bag_classifier/weights/best.pt` + `fill_and_bag_classifier/calibration.json` | `cart_classifier_multihead/weights/best.pt` |
| md5 | quality `0548d26d4cab09ef49d9e553289a985a`; fill/bag `e3d1991a12876f2ad142398a8e16deb4` | `26e10f33df85eb7683d86e502c05411f` |
| Added | 2026-04-30 (commit 4c935f6) | 2026-11-12 |
| Models | Two MobileNetV3-large nets: quality (valid / unclear), then fill (empty / partial / full) + bag (bagged / unbagged) on valid carts | One MobileNetV3-large net with five heads: quality, fill, bag, child (yes / no), stack (single / stacked) |
| Input crop | The exact cart box | The cart box padded top 25%, sides 10%, bottom 5% (clipped to the frame) |
| Calibration | Temperature scaled (quality T 0.78, fill T 0.62, bag T 0.61) | None (softmax at T 1.0) |
| Training labels | Old human-labelled crops (5% padded) | `cart_attr_labels` waves 01-10 on training cameras only: fill / bag kept only where one Sonnet pass and v1 agree; quality / child / stack from Sonnet; plus 93 user-labelled tiles (user gold batch 1). Warm start from the v1 fill/bag backbone. Epoch 12 of 25, chosen on validation. |

## How they compare

Measured 2026-11-12 on 194 tiles the dataset owner labelled by hand: 96 random tiles (user batch 2) plus 98 random
tiles from held-out test cameras (user_test3). Neither model trained on any of them. Accuracy on the tiles the owner
could judge:

| Head | v1 | v2 |
|---|---|---|
| fill | 132/171 = 0.772 | 134/171 = 0.784 |
| bag | 128/167 = 0.766 | 130/167 = 0.778 |
| quality | 151/194 = 0.778 | 172/194 = 0.887 |
| child | (none) | 190/194 (only 1 real child in the set) |
| stack | (none) | 191/194 |

On the 98 held-out-camera tiles alone, v2 leads by more: fill 65/87 against 62/87, bag 64/86 against 58/86.

What this means:
- Fill and bag are a statistical tie. Where the two disagree, v2 is right 14 times and v1 12 times on fill, and 17
  against 15 on bag. Both are well under the 0.90 target.
- Quality is a clear improvement. v1's gate rejected 41 of 172 carts the owner called valid; v2 rejects 10. So many
  more carts get a fill / bag reading at all.
- The two make different mistakes. v1 mostly calls empty carts partial (often a child alone in the cart). v2 mostly
  confuses partial with full and misses bags it should see.
- v2's confidences are not temperature scaled, so the confidence-weighted vote in `engine/scoring.py` sees a softer
  scale than with v1. Labels are unaffected.
- child / stack come out as extra keys on each classification (`child_in_cart`, `child_conf`, `stacked`,
  `stack_conf`). Nothing in the engine decides on them yet.

Clip spot check (2026-11-12): 9 held-out-camera clips (3 Confirmed, 3 Suspicious, 3 Not_A_Pushout), each run with
v1 and then v2, same placement and settings. 7 of 9 clip verdicts unchanged and all 3 PUSHOUT ALERTs kept. One
Not_A_Pushout clip improved (NEEDS_REVIEW / MEDIUM to SKIP / LOW: its cart's bag went unbagged to bagged). One
Confirmed clip dropped HIGH to MEDIUM (its cart read partial instead of full). Bag flipped in 5 carts, in both
directions, matching the tile test where bag is the weakest head. Nine clips show what changes, not which is better.

Engine check: for 40 held-out-camera tiles, the engine's own crop of the source frame matched the training crop box
40/40, and its labels matched the offline scores 40/40.

## Where it came from

Training code and data: `D:\gatekeeper_projects\cart_attr_labels` (`tools/p4_train.py`, run `models/mh_w01-10`).
The checkpoint carries its provenance (`version`, `source`, `source_md5`, `trained_on`, `epoch`, `val`).
