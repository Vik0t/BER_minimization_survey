from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "airi_video"
VIDEO_PATH = ROOT / "reports" / "airi_application_video.mp4"
AUDIO_PATH = OUT_DIR / "voice.aiff"
CONCAT_PATH = OUT_DIR / "concat.txt"

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

W, H = 1920, 1080


NARRATION = """Здравствуйте, меня зовут Виктор. Я хочу попасть на Лето с AIRI, потому что для меня это возможность поработать в среде, где идеи проверяются не только словами, но и аккуратными экспериментами.

Сейчас я занимаюсь нейросетевой эквализацией шестнадцати-QAM сигнала. Мне интересно направление Efficient DL: как получить выигрыш качества, не потеряв вычислительную эффективность.

Моя идея в Research Proposal основана на статье KAN: Kolmogorov-Arnold Networks, ICLR 2025. Я рассматриваю KAN как альтернативу MLP для нелинейной компенсации канала. Вместо фиксированных активаций KAN обучает одномерные функции на ребрах, и это может быть полезно для гладких физических нелинейностей.

В пилотных экспериментах базовый BER был около девяти и семи десятых на десять в минус третьей. MLP снизил его до двух и трех десятых на десять в минус третьей, а KAN-классификатор -- до восьми и девяти десятых на десять в минус четвертой. Но KAN заметно тяжелее, поэтому главный вопрос -- не просто получить лучший BER, а найти хороший компромисс BER и сложности.

На Школе я хочу довести эту идею до более строгого исследования: сравнить hidden dimension, размер окна, grid size, порядок сплайна, число слоев, а затем попробовать pruning или FastKAN. Для моей карьеры это важный шаг от инженерных экспериментов к полноценной исследовательской работе.

Я хочу приехать не только учиться, но и принести проект, который можно критиковать, улучшать и превращать в понятный научный результат. Спасибо."""


SLIDES = [
    {
        "title": "Лето с AIRI 2026",
        "subtitle": "Почему я хочу участвовать",
        "bullets": ["EFFICIENTDL", "нейросетевая эквализация", "KAN vs MLP для 16QAM"],
    },
    {
        "title": "Мотивация",
        "subtitle": "Мне важна исследовательская среда",
        "bullets": [
            "идеи проверяются экспериментами",
            "есть сильная обратная связь",
            "можно превратить проект в научный результат",
        ],
    },
    {
        "title": "Карьера и развитие",
        "subtitle": "От инженерного прототипа к исследованию",
        "bullets": [
            "формулировать гипотезы",
            "строить честный train/val/test протокол",
            "сравнивать качество, скорость и сложность",
        ],
    },
    {
        "title": "Research Proposal",
        "subtitle": "KAN для нелинейной эквализации 16QAM",
        "bullets": [
            "статья: KAN, ICLR 2025 Oral",
            "направление: EFFICIENTDL",
            "идея: BER-aware KAN вместо обычного MLP",
        ],
    },
    {
        "title": "Пилотные результаты",
        "subtitle": "BER на тестовых файлах",
        "bullets": [
            "baseline: 9.71e-3",
            "MLP: 2.27e-3",
            "KAN-classifier: 8.90e-4",
        ],
        "chart": True,
    },
    {
        "title": "Главный вопрос",
        "subtitle": "KAN лучше по BER, но тяжелее",
        "bullets": [
            "MLP быстрее примерно на порядок",
            "KAN дает сильный выигрыш качества",
            "цель: оптимальная точка BER vs Complexity",
        ],
    },
    {
        "title": "План на Школу",
        "subtitle": "Довести идею до строгого исследования",
        "bullets": [
            "grid, spline order, hidden dim, окно, layers",
            "pruning или FastKAN",
            "компактный KAN-эквалайзер",
        ],
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def draw_gradient(draw: ImageDraw.ImageDraw) -> None:
    for y in range(H):
        t = y / (H - 1)
        r = int(11 + 20 * t)
        g = int(34 + 38 * t)
        b = int(49 + 50 * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def wrap_text(text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_block(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt, fill, max_width, line_gap=12) -> int:
    x, y = xy
    for line in wrap_text(text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def draw_chart(draw: ImageDraw.ImageDraw) -> None:
    labels = ["Baseline", "MLP", "KAN"]
    values = [9.71e-3, 2.27e-3, 0.89e-3]
    colors = [(234, 179, 8), (56, 189, 248), (52, 211, 153)]
    x0, y0 = 1110, 360
    bar_w, max_h = 150, 430
    max_val = max(values)
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        x = x0 + i * 210
        h = int(max_h * (val / max_val))
        y = y0 + max_h - h
        draw.rounded_rectangle((x, y, x + bar_w, y0 + max_h), radius=18, fill=color)
        draw.text((x - 5, y0 + max_h + 30), label, font=font(34, True), fill=(236, 253, 245))
        draw.text((x - 12, y - 54), f"{val:.2e}", font=font(32, True), fill=(255, 255, 255))
    draw.line((x0 - 40, y0 + max_h, x0 + 3 * 210 - 20, y0 + max_h), fill=(180, 220, 230), width=4)


def make_slide(idx: int, slide: dict) -> Path:
    img = Image.new("RGB", (W, H), (10, 24, 34))
    draw = ImageDraw.Draw(img)
    draw_gradient(draw)

    # Decorative signal-like curves.
    for k, color in enumerate([(45, 212, 191), (56, 189, 248), (250, 204, 21)]):
        points = []
        for x in range(-40, W + 40, 12):
            y = int(820 + 42 * math.sin((x / 115) + k * 1.9) + 23 * math.sin((x / 53) + k))
            points.append((x, y + k * 34))
        draw.line(points, fill=color + (160,) if len(color) == 3 else color, width=5)

    draw.rounded_rectangle((78, 70, 1842, 1010), radius=46, outline=(125, 211, 252), width=3)
    draw.text((130, 130), slide["title"], font=font(76, True), fill=(240, 253, 250))
    draw_text_block(draw, slide["subtitle"], (132, 235), font(46), (165, 243, 252), 970)

    y = 390
    for bullet in slide["bullets"]:
        draw.ellipse((145, y + 14, 168, y + 37), fill=(52, 211, 153))
        y = draw_text_block(draw, bullet, (190, y), font(42), (226, 232, 240), 830, line_gap=10) + 28

    if slide.get("chart"):
        draw_chart(draw)
    else:
        draw.rounded_rectangle((1150, 355, 1660, 705), radius=36, fill=(15, 118, 110), outline=(94, 234, 212), width=2)
        draw.text((1215, 410), "BER", font=font(86, True), fill=(240, 253, 250))
        draw.text((1215, 515), "vs", font=font(52, True), fill=(204, 251, 241))
        draw.text((1215, 585), "Complexity", font=font(54, True), fill=(240, 253, 250))

    draw.text((132, 940), "Research Proposal: KAN for 16QAM equalization", font=font(28), fill=(148, 163, 184))
    path = OUT_DIR / f"slide_{idx:02d}.png"
    img.save(path)
    return path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide_paths = [make_slide(i, slide) for i, slide in enumerate(SLIDES, start=1)]

    narration_path = OUT_DIR / "narration.txt"
    narration_path.write_text(NARRATION, encoding="utf-8")

    run(["say", "-v", "Milena", "-r", "176", "-f", str(narration_path), "-o", str(AUDIO_PATH)])
    audio_duration = min(probe_duration(AUDIO_PATH), 119.0)
    durations = [13, 14, 14, 17, 20, 17, 14]
    scale = audio_duration / sum(durations)
    durations = [max(4.0, d * scale) for d in durations]

    lines = []
    for path, duration in zip(slide_paths, durations):
        lines.append(f"file '{path}'")
        lines.append(f"duration {duration:.3f}")
    lines.append(f"file '{slide_paths[-1]}'")
    CONCAT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw_video = OUT_DIR / "slides.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(CONCAT_PATH),
            "-vf",
            "fps=30,format=yuv420p",
            "-r",
            "30",
            str(raw_video),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video),
            "-i",
            str(AUDIO_PATH),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(VIDEO_PATH),
        ]
    )
    print(VIDEO_PATH)
    print(f"duration={probe_duration(VIDEO_PATH):.2f}s")


if __name__ == "__main__":
    main()
