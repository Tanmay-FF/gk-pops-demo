# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""Real person/cart box geometry from the 1764099569430 OUTSIDE clip.

Every box here was read out of that run's own tracking JSON, so the numbers are
what the pipeline actually saw — not a hand-drawn approximation of it. They exist
because the linker's association bug is a GEOMETRY bug, and a synthetic fixture
with tidy boxes cannot reproduce it: what breaks the linker is a scene with real
depth, where a person metres away from a cart still overlaps it in 2D.

WHY THESE SIX CASES
-------------------
Two of them are the mislinks that cost the clip its pushout (see
docs/missed_pushout_fix_plan.md):

  * P4 -> C1  a person in the extreme foreground, truncated by the bottom frame
              edge, whose feet land 151 px BELOW the cart's base. Linked anyway,
              at IoU 0.05, and it held the link from frame 68 to 109 — which is
              what kept C1's real handler from ever being considered.
  * P2 -> C3  the same error in the other direction: C3 is a cart in the far
              background, seen through the doorway, and P2 is 170 px in FRONT of
              its base. Linked at 2D distance 101 px.

Four of them are links that are CORRECT and must survive any gate added to fix
the two above. Three matter especially:

  * P2 -> C1 at frame 24 is the true handler with his lower body OCCLUDED BY THE
              BASKET he is leaning over, so his box ends 210 px above the cart's
              own base. A naive "feet must be near the cart's base" test rejects
              him — the very link the fix exists to enable. What separates him
              from P4 is not distance from the base but WHICH SIDE: his box ends
              INSIDE the cart's vertical extent, P4's ends far below it.
  * P4 -> C4 is the same P4, foreground, correctly owning the foreground cart.
              A gate that just distrusts foreground people breaks this.
  * P3 -> C3 is C3's real owner while C3 is still near the door.

Boxes are (x1, y1, x2, y2), exactly as the linker receives them.
"""

#: (label, frame, person_box, cart_box, should_link, why)
LINK_CASES = (
    (
        "P4->C1 foreground body clipping a background cart", 68,
        (613, 532, 852, 718), (661, 321, 827, 567), False,
        "feet 151 px below the cart's base (0.61x cart height); bbox is also "
        "truncated by the bottom frame edge at 718 of 719, so its ground point "
        "is not even known. IoU 0.05. This is the mislink that cost the pushout.",
    ),
    (
        "P2->C3 person in front of a cart seen through the doorway", 68,
        (772, 209, 905, 558), (870, 226, 939, 388), False,
        "feet 170 px in front of the cart's base (1.05x cart height). 2D "
        "distance 101 px reads as adjacent; in the scene they are metres apart.",
    ),
    (
        "P2->C1 true handler, lower body occluded by the basket", 24,
        (607, 211, 739, 369), (605, 331, 744, 579), True,
        "box ends 210 px ABOVE the cart's base because he is leaning over it — "
        "but inside the cart's own vertical extent (331..579), not below it. "
        "Must survive: this is the link whose absence broke the clip.",
    ),
    (
        "P2->C1 same pair once he straightens up", 68,
        (772, 209, 905, 558), (661, 321, 827, 567), True,
        "feet 9 px from the cart's base. The unambiguous version of the case "
        "above.",
    ),
    (
        "P4->C4 foreground person, foreground cart", 68,
        (613, 532, 852, 718), (643, 502, 855, 706), True,
        "the same P4 whose link to C1 is wrong, correctly owning the cart that "
        "is actually with him. A gate that distrusts foreground people breaks "
        "this one.",
    ),
    (
        "P3->C3 real owner while the cart is still near the door", 24,
        (808, 223, 944, 576), (780, 292, 852, 523), True,
        "feet 53 px past the base of a 231 px cart (0.23x). Legitimate.",
    ),
)


def foot_delta(person_box, cart_box):
    """Signed distance from the cart's base to the person's, in pixels.

    Positive means the person's box ends BELOW the cart's base — nearer the
    camera. Negative means above it — further away, or occluded by the cart.
    """
    return person_box[3] - cart_box[3]


def cart_height(cart_box):
    return cart_box[3] - cart_box[1]


def ends_inside_cart(person_box, cart_box):
    """Does the person's box end within the cart's own vertical extent?

    True for someone leaning into a basket, whose legs the cart hides. This is
    the case that separates the occluded true handler from a foreground body: the
    handler's box ends INSIDE the cart, the foreground body's ends below it.
    """
    return cart_box[1] <= person_box[3] <= cart_box[3]
