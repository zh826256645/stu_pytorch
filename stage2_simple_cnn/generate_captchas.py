"""生成与当前四位验证码尺寸和干扰风格相近的合成图片。

生成器用于创建固定且可复现的独立合成测试集。它不能替代真实来源的新数据；
一旦使用生成结果评估模型，就不应再根据该批图片调整模型或生成参数。
"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from datasets import SUPPORTED_IMAGE_SUFFIXES

DEFAULT_CHARACTERS = "2345678abcdefgmnpwxy"
DEFAULT_COUNT = 400
DEFAULT_SEED = 20250310
DEFAULT_OUTPUT_DIR = Path("data/independent_test")
DEFAULT_WIDTH = 180
DEFAULT_HEIGHT = 100
FONT_SUFFIXES = {".otf", ".ttc", ".ttf"}
IGNORED_FONT_NAME_PARTS = (
    "braille",
    "emoji",
    "georgian",
    "icon",
    "ornament",
    "symbol",
    "wingding",
)
PREFERRED_FONT_NAME_PARTS = (
    "american typewriter",
    "arial",
    "avenir",
    "baskerville",
    "bodoni 72",
    "calibri",
    "cambria",
    "candara",
    "charter",
    "cochin",
    "comic sans",
    "courier",
    "dejavu",
    "didot",
    "futura",
    "geneva",
    "georgia",
    "gill sans",
    "gillsans",
    "helvetica",
    "hoefler text",
    "impact",
    "iowan old style",
    "liberation",
    "lucidagrande",
    "markerfelt",
    "menlo",
    "microsoft sans serif",
    "monaco",
    "newyork",
    "noteworthy",
    "optima",
    "palatino",
    "roboto",
    "sfcompact",
    "sfns",
    "tahoma",
    "times",
    "trebuchet",
    "ubuntu",
    "verdana",
)
FOREGROUND_COLORS = (
    (10, 10, 10),
    (75, 75, 75),
    (215, 10, 20),
    (10, 45, 220),
    (15, 155, 55),
    (245, 220, 0),
    (135, 20, 175),
)


@dataclass(frozen=True)
class GeneratedCaptcha:
    """保存单张合成验证码的可复现元数据。"""

    file: str
    label: str
    sha256: str
    font: str
    font_size: int
    foreground: tuple[int, int, int]
    background_style: str
    grid: bool
    lines_before_text: int
    lines_after_text: int
    blur_radius: float


def common_font_directories() -> tuple[Path, ...]:
    """返回 macOS、Linux 和 Windows 上常见的字体目录。"""
    directories = [
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]
    windows_directory = os.environ.get("WINDIR")
    if windows_directory:
        directories.append(Path(windows_directory) / "Fonts")
    return tuple(directories)


def font_supports_characters(path: Path, characters: str) -> bool:
    """过滤无法正常绘制目标 ASCII 字符的字体。"""
    try:
        font = ImageFont.truetype(str(path), size=48)
        glyph_signatures = []
        for character in characters:
            glyph_mask = font.getmask(character)
            if glyph_mask.getbbox() is None:
                return False
            glyph_signatures.append((glyph_mask.size, bytes(glyph_mask)))
        # 缺少拉丁字形的字体常把所有字符渲染为同一个占位符。
        return len(set(glyph_signatures)) == len(characters)
    except (OSError, ValueError):
        return False


def is_preferred_font(path: Path) -> bool:
    """限制自动发现结果为常见拉丁字体，避免符号和装饰字体。"""
    normalized_name = path.name.lower().replace("-", " ").replace("_", " ")
    return any(part in normalized_name for part in PREFERRED_FONT_NAME_PARTS)


def discover_font_paths(
    requested_paths: list[Path] | None,
    *,
    characters: str,
) -> list[Path]:
    """发现可用字体；用户路径优先，否则扫描系统常见字体目录。"""
    roots = requested_paths or list(common_font_directories())
    candidates = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() in FONT_SUFFIXES:
            candidates.add(root.resolve())
        elif root.is_dir():
            for suffix in FONT_SUFFIXES:
                candidates.update(path.resolve() for path in root.rglob(f"*{suffix}"))
                candidates.update(
                    path.resolve() for path in root.rglob(f"*{suffix.upper()}")
                )

    filtered = [
        path
        for path in sorted(candidates)
        if not any(part in path.name.lower() for part in IGNORED_FONT_NAME_PARTS)
        and (requested_paths is not None or is_preferred_font(path))
        and font_supports_characters(path, characters)
    ]
    if not filtered:
        raise ValueError(
            "没有找到能够绘制目标字符的 TTF/OTF/TTC 字体。"
            "请通过 --font-path 指定字体文件或目录。"
        )
    return filtered


def generate_balanced_labels(
    count: int,
    characters: str,
    rng: random.Random,
) -> list[str]:
    """生成唯一标签，并让每个位置的字符数量尽可能均衡。"""
    if count <= 0:
        raise ValueError("count 必须大于 0")
    if len(characters) < 2 or len(set(characters)) != len(characters):
        raise ValueError("characters 至少需要两个不重复字符")
    if count > len(characters) ** 4:
        raise ValueError("count 超过四位字符能够组成的唯一标签数量")

    character_count = len(characters)
    complete_rounds, remainder = divmod(count, character_count)
    position_alphabets = []
    for _position in range(4):
        shuffled_characters = list(characters)
        rng.shuffle(shuffled_characters)
        position_alphabets.append(shuffled_characters)

    labels = []
    for round_index in range(complete_rounds + (1 if remainder else 0)):
        first_indices = list(range(character_count))
        if round_index == complete_rounds and remainder:
            first_indices = rng.sample(first_indices, remainder)

        second_offset = round_index % character_count
        third_offset = (round_index // character_count) % character_count
        fourth_offset = (round_index // (character_count**2)) % character_count
        for first_index in first_indices:
            position_indices = (
                first_index,
                (first_index + second_offset) % character_count,
                (first_index + third_offset) % character_count,
                (first_index + fourth_offset) % character_count,
            )
            labels.append(
                "".join(
                    position_alphabets[position][character_index]
                    for position, character_index in enumerate(position_indices)
                )
            )

    rng.shuffle(labels)
    if len(labels) != count or len(set(labels)) != count:
        raise RuntimeError("验证码标签生成结果不满足数量或唯一性约束")
    return labels


def create_background(
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[Image.Image, str]:
    """生成灰白色纯色或渐变背景，并叠加轻微噪点。"""
    base = rng.randint(188, 242)
    style = rng.choices(
        ("solid", "vertical-gradient", "horizontal-gradient"),
        weights=(45, 30, 25),
        k=1,
    )[0]
    image = Image.new("RGB", (width, height), (base, base, base))
    pixels = image.load()

    if style != "solid":
        delta = rng.randint(-25, 25)
        for y in range(height):
            for x in range(width):
                progress = y / max(1, height - 1)
                if style == "horizontal-gradient":
                    progress = x / max(1, width - 1)
                value = max(165, min(250, round(base + delta * progress)))
                pixels[x, y] = (value, value, value)

    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(80, 360)):
        x = rng.randrange(width)
        y = rng.randrange(height)
        offset = rng.randint(-24, 24)
        current = pixels[x, y][0]
        value = max(145, min(255, current + offset))
        draw.point((x, y), fill=(value, value, value))
    return image.convert("RGBA"), style


def draw_grid(
    image: Image.Image,
    rng: random.Random,
    width: int,
    height: int,
) -> bool:
    """小概率加入类似历史困难样本的彩色网格干扰。"""
    if rng.random() >= 0.13:
        return False

    draw = ImageDraw.Draw(image, "RGBA")
    spacing_x = rng.randint(7, 12)
    spacing_y = rng.randint(7, 12)
    offset_x = rng.randrange(spacing_x)
    offset_y = rng.randrange(spacing_y)
    vertical_color = rng.choice(((230, 0, 20, 185), (0, 40, 225, 185)))
    horizontal_color = (
        (0, 40, 225, 185)
        if vertical_color[0] > vertical_color[2]
        else (230, 0, 20, 185)
    )
    for x in range(offset_x, width, spacing_x):
        draw.line((x, 0, x, height), fill=vertical_color, width=1)
    for y in range(offset_y, height, spacing_y):
        draw.line((0, y, width, y), fill=horizontal_color, width=1)
    return True


def interference_color(
    foreground: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """大多数干扰线沿用文字颜色，少量使用对比色。"""
    if rng.random() < 0.72:
        return (*foreground, rng.randint(145, 235))
    color = rng.choice(FOREGROUND_COLORS)
    return (*color, rng.randint(110, 200))


def draw_interference_lines(
    image: Image.Image,
    rng: random.Random,
    foreground: tuple[int, int, int],
    count: int,
) -> None:
    """绘制折线、斜线或弧线干扰。"""
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for _ in range(count):
        color = interference_color(foreground, rng)
        line_width = rng.randint(1, 3)
        if rng.random() < 0.72:
            points = [
                (
                    rng.randint(-10, width + 10),
                    rng.randint(5, height - 5),
                )
                for _ in range(rng.randint(2, 4))
            ]
            points.sort(key=lambda point: point[0])
            draw.line(points, fill=color, width=line_width, joint="curve")
        else:
            x1 = rng.randint(-30, width - 20)
            y1 = rng.randint(-30, height - 20)
            x2 = x1 + rng.randint(55, 150)
            y2 = y1 + rng.randint(35, 110)
            start_angle = rng.randint(0, 150)
            draw.arc(
                (x1, y1, x2, y2),
                start=start_angle,
                end=start_angle + rng.randint(90, 230),
                fill=color,
                width=line_width,
            )


def render_character(
    character: str,
    font_path: Path,
    font_size: int,
    foreground: tuple[int, int, int],
    rng: random.Random,
) -> Image.Image:
    """把单个字符绘制到透明图层，并施加轻微伸缩与旋转。"""
    font = ImageFont.truetype(str(font_path), size=font_size)
    stroke_width = rng.choice((0, 0, 1, 1, 2))
    bounding_box = font.getbbox(character, stroke_width=stroke_width)
    character_width = bounding_box[2] - bounding_box[0]
    character_height = bounding_box[3] - bounding_box[1]
    margin = 10
    layer = Image.new(
        "RGBA",
        (character_width + margin * 2, character_height + margin * 2),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(layer, "RGBA")
    origin = (margin - bounding_box[0], margin - bounding_box[1])

    if rng.random() < 0.58:
        shadow_offset = rng.randint(1, 4)
        shadow_alpha = rng.randint(55, 130)
        draw.text(
            (origin[0] + shadow_offset, origin[1] + shadow_offset),
            character,
            font=font,
            fill=(30, 30, 30, shadow_alpha),
            stroke_width=stroke_width,
            stroke_fill=(30, 30, 30, shadow_alpha),
        )

    draw.text(
        origin,
        character,
        font=font,
        fill=(*foreground, 255),
        stroke_width=stroke_width,
        stroke_fill=(*foreground, 255),
    )

    horizontal_scale = rng.uniform(0.82, 1.18)
    scaled_width = max(1, round(layer.width * horizontal_scale))
    layer = layer.resize((scaled_width, layer.height), Image.Resampling.BICUBIC)
    return layer.rotate(
        rng.uniform(-12, 12),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )


def calculate_sha256(path: Path) -> str:
    """计算生成文件哈希，用于冻结和复核测试集。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_one_captcha(
    label: str,
    font_path: Path,
    output_path: Path,
    rng: random.Random,
    *,
    width: int,
    height: int,
) -> GeneratedCaptcha:
    """生成单张验证码并返回其样式元数据。"""
    image, background_style = create_background(width, height, rng)
    foreground = rng.choice(FOREGROUND_COLORS)
    grid = draw_grid(image, rng, width, height)
    lines_before_text = rng.randint(0, 3)
    draw_interference_lines(
        image,
        rng,
        foreground,
        lines_before_text,
    )

    font_size = rng.randint(48, 66)
    slot_width = width / 4
    for position, character in enumerate(label):
        layer = render_character(
            character,
            font_path,
            font_size + rng.randint(-5, 5),
            foreground,
            rng,
        )
        center_x = (position + 0.5) * slot_width + rng.uniform(-5, 5)
        x = round(center_x - layer.width / 2)
        y = round(rng.uniform(8, max(9, height - layer.height - 4)))
        image.paste(layer, (x, y), layer)

    lines_after_text = rng.randint(0, 3)
    draw_interference_lines(
        image,
        rng,
        foreground,
        lines_after_text,
    )

    blur_radius = rng.choice((0.0, 0.0, 0.25, 0.4, 0.6))
    if blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    image.convert("RGB").save(output_path, format="PNG", optimize=True)

    return GeneratedCaptcha(
        file=output_path.name,
        label=label,
        sha256=calculate_sha256(output_path),
        font=str(font_path),
        font_size=font_size,
        foreground=foreground,
        background_style=background_style,
        grid=grid,
        lines_before_text=lines_before_text,
        lines_after_text=lines_after_text,
        blur_radius=blur_radius,
    )


def remove_existing_generated_files(output_dir: Path) -> None:
    """仅删除目标目录中的图片和本生成器清单。"""
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            path.unlink()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def generate_dataset(
    *,
    output_dir: Path,
    count: int,
    seed: int,
    characters: str,
    font_paths: list[Path],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    replace_existing: bool = False,
) -> list[GeneratedCaptcha]:
    """生成完整数据集，并写出包含哈希与样式参数的清单。"""
    if width < 80 or height < 40:
        raise ValueError("图片尺寸过小，无法稳定绘制四位验证码")
    if seed < 0:
        raise ValueError("seed 必须大于等于 0")
    if not font_paths:
        raise ValueError("font_paths 不能为空")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if existing_images and not replace_existing:
        raise ValueError(
            f"输出目录已有 {len(existing_images)} 张图片。"
            "为保护已经冻结的测试集，默认拒绝覆盖；如确定重建，请添加 "
            "--replace-existing。"
        )
    if replace_existing:
        remove_existing_generated_files(output_dir)

    rng = random.Random(seed)
    labels = generate_balanced_labels(count, characters, rng)
    records = []
    for index, label in enumerate(labels, start=1):
        font_path = rng.choice(font_paths)
        output_path = output_dir / f"{label}.png"
        records.append(
            generate_one_captcha(
                label,
                font_path,
                output_path,
                rng,
                width=width,
                height=height,
            )
        )
        if index % 50 == 0 or index == count:
            print(f"已生成 {index}/{count} 张")

    manifest = {
        "version": 1,
        "generator": "stage2_simple_cnn.generate_captchas",
        "seed": seed,
        "count": count,
        "width": width,
        "height": height,
        "characters": characters,
        "fonts": [str(path) for path in font_paths],
        "files": [asdict(record) for record in records],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成四位合成验证码图片")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--characters", default=DEFAULT_CHARACTERS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--font-path",
        type=Path,
        action="append",
        default=None,
        help="字体文件或目录；可重复传入，默认扫描系统字体",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="删除目标目录已有图片后重新生成；默认保护现有测试集",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    font_paths = discover_font_paths(
        args.font_path,
        characters=args.characters,
    )
    print(f"可用字体：{len(font_paths)} 个")
    print(
        f"输出目录：{args.output_dir} | 数量：{args.count} | "
        f"随机种子：{args.seed} | 尺寸：{args.width}x{args.height}"
    )
    records = generate_dataset(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        characters=args.characters,
        font_paths=font_paths,
        width=args.width,
        height=args.height,
        replace_existing=args.replace_existing,
    )
    print(f"生成完成：{len(records)} 张 | 清单：{args.output_dir / 'manifest.json'}")
    print("请冻结该批图片；评估后不要再根据其错误调整模型或生成参数。")


if __name__ == "__main__":
    main()
