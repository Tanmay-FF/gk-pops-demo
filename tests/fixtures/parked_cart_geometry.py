# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Real geometry for the parked cart on the 1764200318790 INSIDE clip.

Every number here was read out of that run's own tracking JSON
(1764200318790_B8A44FA1E0F1-medium_tracking.json), so it is what the pipeline
actually saw. 1280x720, 20 fps, 411 frames.

WHAT THE CLIP SHOWS
-------------------
C1 is an empty shopping cart parked just inside the entrance, bottom-left of
frame. It is furniture: over all 411 frames its centroid stays inside x
244.5-246.5, y 511.0-515.0 — a 4.5 px sweep in 20.5 seconds.

P4 is a shopper walking INTO the store, crossing the foreground from bottom-left
to upper-right. Her box is truncated by the bottom frame edge (y2 = 718-719 of
720) for the whole crossing, so she is much nearer the camera than the cart. She
clips the corner of C1's box for six frames — 127 to 132 inclusive, IoU 0.039 to
0.064 — and by frame 135 she has no overlap with it at all.

The linker gave her the cart at frame 132 and never took it back: the takeover
rule needs a THIRD person overlapping the cart, and nobody else ever touched it.
She kept it for the remaining 279 frames, and when she left the doorway at frame
206 the cart became an ABANDONED CART at 65 with her recorded as its owner.

WHY THE EXISTING GATES ALL PASSED HER
-------------------------------------
  * IoU:            peak 0.064, over LINK_MIN_IOU (0.02).
  * Ground plane:   mean foot ratio +0.335 over the six frames, inside
                    LINK_GROUND_BAND (0.45). This clip is the counter-example to
                    the 1764099569430 one: there the mislinked foreground body
                    read +0.64 because the cart was far away, here the cart is
                    ALSO in the foreground, so the same error reads +0.34 and the
                    ground test cannot see it.
  * Co-movement:    the cart is permanently static, so the static_a_ok exemption
                    ("parked cart, person working at it = loading") was
                    permanently open to anyone who overlapped it at all.
  * Threshold:      she was the only candidate, so six frames of overlap met
                    LINK_CONFIRM_FRAMES.

Boxes are (x1, y1, x2, y2) and centroids (cx, cy), exactly as the linker receives
them.
"""

#: C1's box and centroid while P4 crosses it. Frames 120-139, real values.
CART_BY_FRAME = {
    120: ((164, 390, 326, 638), (245.0, 514.5)),
    121: ((164, 390, 326, 638), (245.0, 514.5)),
    122: ((164, 390, 326, 638), (245.0, 514.5)),
    123: ((164, 390, 326, 638), (245.0, 514.5)),
    124: ((164, 391, 326, 638), (245.0, 514.5)),
    125: ((164, 391, 326, 638), (245.0, 514.5)),
    126: ((164, 391, 326, 634), (245.0, 512.5)),
    127: ((164, 391, 326, 633), (245.0, 512.0)),
    128: ((164, 391, 326, 634), (245.0, 512.5)),
    129: ((164, 391, 326, 634), (245.0, 512.5)),
    130: ((164, 391, 327, 636), (245.5, 513.5)),
    131: ((164, 391, 326, 637), (245.0, 514.0)),
    132: ((164, 391, 326, 638), (245.0, 514.5)),
    133: ((164, 391, 326, 638), (245.0, 514.5)),
    134: ((164, 391, 326, 638), (245.0, 514.5)),
    135: ((164, 391, 327, 638), (245.5, 514.5)),
    136: ((164, 391, 326, 638), (245.0, 514.5)),
    137: ((164, 391, 326, 638), (245.0, 514.5)),
    138: ((164, 391, 326, 638), (245.0, 514.5)),
    139: ((164, 391, 326, 638), (245.0, 514.5)),
}

#: The box and centroid the cart holds for the rest of the run. Every frame
#: outside CART_BY_FRAME is this, to within the 4.5 px sweep documented above.
CART_PARKED = ((164, 391, 326, 638), (245.0, 514.5))

#: The diagonal of the box C1's centroids sweep over the WHOLE clip. Compare
#: against LINK_STATIC_SPREAD_PX: this cart is parked by any reading.
CART_CENTROID_SPAN_PX = 4.47

#: P4's box and centroid, frames 127-139 — from her first detection through the
#: end of any contact with the cart. y2 is the frame edge throughout.
P4_BY_FRAME = {
    127: ((223, 613, 375, 718), (299.0, 665.5)),
    128: ((238, 600, 388, 718), (313.0, 659.0)),
    129: ((248, 589, 393, 718), (320.5, 653.5)),
    130: ((265, 577, 416, 718), (340.5, 647.5)),
    131: ((277, 568, 430, 718), (353.5, 643.0)),
    132: ((293, 558, 450, 718), (371.5, 638.0)),
    133: ((304, 551, 461, 718), (382.5, 634.5)),
    134: ((321, 541, 476, 718), (398.5, 629.5)),
    135: ((333, 532, 487, 718), (410.0, 625.0)),
    136: ((348, 519, 500, 719), (424.0, 619.0)),
    137: ((358, 508, 509, 719), (433.5, 613.5)),
    138: ((369, 495, 538, 719), (453.5, 607.0)),
    139: ((379, 486, 558, 719), (468.5, 602.5)),
}

#: The frame the link was established on in the faulty run, and the frames P4
#: overlapped the cart at all.
LINKED_ON_FRAME = 132
OVERLAP_FRAMES = (127, 128, 129, 130, 131, 132, 133, 134)

#: Where P4 is later in the run, still visible and nowhere near the cart. Her
#: track ends at 206; the cart was scored ABANDONED CART from 237.
P4_WALKING_AWAY = {
    166: ((660, 291, 779, 578), (719.5, 434.5)),
    206: ((924, 217, 970, 413), (947.0, 315.0)),
}
P4_LAST_FRAME = 206
ABANDONED_FROM_FRAME = 237
