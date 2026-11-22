"""Render saved terminal outputs as terminal-style PNG screenshots."""
from PIL import Image, ImageDraw, ImageFont
font = None
for f in ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
          "consola.ttf", "Menlo.ttc", "Courier New.ttf"]:
    try:
        font = ImageFont.truetype(f, 15); break
    except OSError:
        pass
font = font or ImageFont.load_default()
for name, cmd in [("customer", "python recommend.py --customer 17850 --k 10"),
                  ("product", "python recommend.py --product 85123A --k 8"),
                  ("coldstart", "python recommend.py --new-basket 22423 21232 22720 --k 8"),
                  ("experiment", "python run_experiment.py data/retail.csv")]:
    src = f"screenshots/raw/out_{name}.txt"
    lines = [f"$ {cmd}"] + open(src).read().rstrip("\n").split("\n")
    w = max(font.getlength(l) for l in lines) + 40
    h = 22 * len(lines) + 60
    img = Image.new("RGB", (int(w), h), "#1e1e1e"); dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, w, 28], fill="#333")
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        dr.ellipse([12 + i * 20, 8, 24 + i * 20, 20], fill=c)
    for i, l in enumerate(lines):
        col = "#7ee787" if i == 0 else ("#79c0ff" if l.isupper() or l.startswith(("Recommended", "Customers", "Similar", "TEST", "NDCG")) else "#e6e6e6")
        dr.text((20, 40 + 22 * i), l, font=font, fill=col)
    img.save(f"screenshots/ss_{name}.png")
    print("saved", name)
