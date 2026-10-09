"""Two-slide v-space deck: model5 from `decipher_models2` (left) vs `decipher_m5` (right).

ADDED 2026-10-08. Checks that swapping model5's one-hot batch input for two `nn.Embedding`
tables (`decipher_m5`) leaves training results unchanged. Sigma 2, seed 3, decipherseed 1-3.
Left = `sweep1008_splitfix` figures (`1008/figs/`), right = `sweep1008_m5` figures
(`1008/figs_decipher_m5/`, from `1008_sweep_splitfix_bifurcation.py --package decipher_m5`).
Slide 1: `decipher_m5` from its own start (same decipherseed, different initial weights).
Slide 2: `decipher_m5` from `decipher_models2`'s start (`1008_m5_matched_start.py`, figures in
`1008/m5_matched_start/figs_decipher_m5/`).
Reuses the template and helpers of `1008_build_splitfix_decks.py`; the template is read, never
written. ``--check`` only re-verifies.
"""

import importlib
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
decks = importlib.import_module("1008_build_splitfix_decks")

SIGMA, SEED, DSEEDS = 2.0, 3, [1, 2, 3]
M5_FIGS = decks.FIGS / "figs_decipher_m5"
M5_MATCHED_FIGS = decks.FIGS / "m5_matched_start" / "figs_decipher_m5"
OUT = decks.WRITEUPS / "1008_vspace_model5_models2_vs_m5_sigma2_seed3_bif.pptx"
LEFT_HDR, RIGHT_HDR = "model5, decipher_models2", "model5, decipher_m5"


def spec():
    """Per slide: (title, subtitle, left_hdr, right_hdr, small_sub, left[3], right[3])."""

    def paths(root):
        return [decks.fig_path(root, "model5", "vspace", SIGMA, SEED, d, "model5") for d in DSEEDS]

    title, left = decks.title_text(SIGMA, SEED), paths(decks.SPLITFIX_FIGS)
    return [
        (title, decks.SUB, LEFT_HDR, RIGHT_HDR, False, left, paths(M5_FIGS)),
        (
            title,
            "decipherseed = 1,2,3 (decipher_m5 started from decipher_models2's weights)",
            LEFT_HDR,
            RIGHT_HDR + ", matched start",
            True,
            left,
            paths(M5_MATCHED_FIGS),
        ),
    ]


def build():
    parts, order, titles = decks.read_template()
    skeleton = parts[titles[decks.title_text(decks.SIGMAS[0], decks.SEEDS[0])]].decode()
    slides = []
    for title, sub, lh, rh, small, left, right in spec():
        for p in left + right:
            assert p.exists(), p
        pics = decks.vspace_pics(left, right)
        xml = decks.slide_xml(skeleton, title, sub, lh, rh, pics, small_sub=small)
        slides.append((xml, [p[0] for p in pics]))
    decks.write_deck(OUT, slides, parts, order)


def check(template_sha_before):
    """Slide count, texts, embedded bytes == source PNGs, template geometry, template unchanged."""
    z = zipfile.ZipFile(OUT)
    assert z.testzip() is None
    pres = z.read("ppt/presentation.xml").decode()
    placed = [
        (decks.COL_X[c], decks.ROW_Y[r], decks.PIC_W, decks.PIC_H)
        for c in (0, 1)
        for r in (0, 1, 2)
    ]
    slides = spec()
    results = [("slide count", len(re.findall(r"<p:sldId ", pres)) == len(slides))]
    for i, (title, sub, lh, rh, _, left, right) in enumerate(slides, 1):
        xml = z.read(f"ppt/slides/slide{i}.xml").decode()
        rels = z.read(f"ppt/slides/_rels/slide{i}.xml.rels").decode()
        rmap = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="\.\./media/(image\d+\.png)"', rels))
        pics = re.findall(
            r'r:embed="(rId\d+)"/>.*?<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"/>',
            xml,
        )
        texts = [t.replace("&amp;", "&") for t in re.findall(r"<a:t>(.*?)</a:t>", xml)]
        results += [
            (f"slide {i} texts", texts == [title, sub, lh, rh]),
            (f"slide {i} 6 pictures", len(pics) == 6),
            (
                f"slide {i} embedded bytes == source PNG sha256",
                all(
                    decks.sha(z.read("ppt/media/" + rmap[rid])) == decks.sha(src.read_bytes())
                    for (rid, *_), src in zip(pics, left + right)
                ),
            ),
            (f"slide {i} template geometry", [tuple(map(int, p[1:])) for p in pics] == placed),
        ]
    results.append(
        ("template sha256 unchanged", decks.sha(decks.TEMPLATE.read_bytes()) == template_sha_before)
    )
    return results


if __name__ == "__main__":
    before = decks.sha(decks.TEMPLATE.read_bytes())
    if "--check" not in sys.argv:
        build()
    results = check(before)
    for name, ok in results:
        print("PASS" if ok else "FAIL", name)
    print(OUT)
    sys.exit(0 if all(ok for _, ok in results) else 1)
