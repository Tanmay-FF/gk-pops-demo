The six demo clips in this folder are tracked in git via LFS
(.gitattributes), so a fresh clone can run the demo without anyone sending a
file separately. Run `git lfs install && git lfs pull` after cloning, or the
clips arrive as small pointer files.

These are real store recordings and they show identifiable people. Do not
redistribute them outside whoever is already entitled to the footage, and do
not add further clips here on the assumption that this folder is fair game:
*.mp4 stays gitignored everywhere else in the repo, and each of these six was
committed as a deliberate decision.

Each clip has a saved zone set in zone_presets/, committed alongside it, and
every one of those records the camera placement it was drawn with. So
`--auto-zones` on a fresh clone gets both the polygons and the angle without
anyone having to know which way the camera faced.

Drop any other clip you want to try in this folder as well - the file picker
lists whatever is here. Extra clips show up as untracked; leave them that way.
