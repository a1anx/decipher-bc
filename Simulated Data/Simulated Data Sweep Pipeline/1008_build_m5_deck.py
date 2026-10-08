"""One-slide v-space deck: model5 from `decipher_models2` (left) vs `decipher_m5` (right).

ADDED 2026-10-08. Checks that swapping model5's one-hot batch input for two `nn.Embedding`
tables (`decipher_m5`) leaves training results unchanged. Sigma 2, seed 3, decipherseed 1-3.
Left = `sweep1008_splitfix` figures (`1008/figs/`), right = `sweep1008_m5` figures
(`1008/figs_decipher_m5/`, from `1008_sweep_splitfix_bifurcation.py --package decipher_m5`).
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
OUT = decks.WRITEUPS / "1008_vspace_model5_models2_vs_m5_sigma2_seed3_bif.pptx"
LEFT_HDR, RIGHT_HDR = "model5, decipher_models2", "model5, decipher_m5"


def spec():
    """(title, subtitle, left_hdr, right_hdr, left[3], right[3]) for the one slide."""
    left = [
        decks.fig_path(decks.SPLITFIX_FIGS, "model5", "vspace", SIGMA, SEED, d, "model5")
        for d in DSEEDS
    ]
    right = [decks.fig_path(M5_FIGS, "model5", "vspace", SIGMA, SEED, d, "model5") for d in DSEEDS]
    return decks.title_text(SIGMA, SEED), decks.SUB, LEFT_HDR, RIGHT_HDR, left, right


def build():
    parts, order, titles = decks.read_template()
    title, sub, lh, rh, left, right = spec()
    for p in left + right:
        assert p.exists(), p
    skeleton = parts[titles[decks.title_text(decks.SIGMAS[0], decks.SEEDS[0])]].decode()
    pics = decks.vspace_pics(left, right)
    xml = decks.slide_xml(skeleton, title, sub, lh, rh, pics)
    decks.write_deck(OUT, [(xml, [p[0] for p in pics])], parts, order)


def check(template_sha_before):
    """Slide count, texts, embedded bytes == source PNGs, template geometry, template unchanged."""
    title, sub, lh, rh, left, right = spec()
    z = zipfile.ZipFile(OUT)
    assert z.testzip() is None
    pres = z.read("ppt/presentation.xml").decode()
    xml = z.read("ppt/slides/slide1.xml").decode()
    rels = z.read("ppt/slides/_rels/slide1.xml.rels").decode()
    rmap = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="\.\./media/(image\d+\.png)"', rels))
    pics = re.findall(
        r'r:embed="(rId\d+)"/>.*?<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"/>', xml
    )
    texts = [t.replace("&amp;", "&") for t in re.findall(r"<a:t>(.*?)</a:t>", xml)]
    placed = [
        (decks.COL_X[c], decks.ROW_Y[r], decks.PIC_W, decks.PIC_H)
        for c in (0, 1)
        for r in (0, 1, 2)
    ]
    return [
        ("slide count", len(re.findall(r"<p:sldId ", pres)) == 1),
        ("texts", texts == [title, sub, lh, rh]),
        ("6 pictures", len(pics) == 6),
        (
            "embedded bytes == source PNG sha256",
            all(
                decks.sha(z.read("ppt/media/" + rmap[rid])) == decks.sha(src.read_bytes())
                for (rid, *_), src in zip(pics, left + right)
            ),
        ),
        ("template geometry", [tuple(map(int, p[1:])) for p in pics] == placed),
        (
            "template sha256 unchanged",
            decks.sha(decks.TEMPLATE.read_bytes()) == template_sha_before,
        ),
    ]


if __name__ == "__main__":
    before = decks.sha(decks.TEMPLATE.read_bytes())
    if "--check" not in sys.argv:
        build()
    results = check(before)
    for name, ok in results:
        print("PASS" if ok else "FAIL", name)
    print(OUT)
    sys.exit(0 if all(ok for _, ok in results) else 1)
