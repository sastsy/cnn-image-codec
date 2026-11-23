"""Build the lab report with ReportLab and Pillow."""
import argparse
import csv
from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='[ФИО студента]')
parser.add_argument('--group', default='[номер группы]')
parser.add_argument('--repo', default='[ссылка на репозиторий с исходным кодом]')
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
result = root / 'results/context-map'
output = root / 'output/pdf/cnn-image-codec-report.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
with (result / 'test.csv').open() as f:
    test = list(csv.DictReader(f))
with (result / 'summary.csv').open() as f:
    summary = list(csv.DictReader(f))
with (result / 'validation.csv').open() as f:
    validation = {(int(x['b']), x['hypothesis']): int(x['bytes']) for x in csv.DictReader(f)}

fonts = Path('/System/Library/Fonts/Supplemental')
font_files = ('Arial.ttf', 'Arial Bold.ttf', 'Courier New.ttf')
if not fonts.is_dir():
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    font_files = ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSansMono.ttf')
for name, filename in zip(('Body', 'Bold', 'Mono'), font_files):
    pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
W, H = 960, 540
PAPER, INK = '#FCFAF5', '#132C33'
MUTED, TEAL, ORANGE, RULE = '#466065', '#075E60', '#A63F23', '#D5DEDA'
WHITE, PALE = '#FFFFFF', '#E9F3EF'
c = Canvas(str(output), pagesize=(W, H), pageCompression=1)
c.setTitle('Контекстное кодирование признаков автокодера')
c.setSubject('Лабораторная работа по сжатию изображений')
c.setAuthor(args.name if not args.name.startswith('[') else '')
page = 0


def text(value, x, y, size=17, font='Body', color=INK, align='left'):
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    width = pdfmetrics.stringWidth(value, font, size)
    start = x if align == 'left' else x - width if align == 'right' else x - width / 2
    assert 0 <= start and start + width <= W + 0.01 and 0 <= y <= H, (value, start, width, y)
    c.drawString(start, H - y, value)


def rect(x, y, w, h, fill):
    c.setFillColor(HexColor(fill))
    c.rect(x, H-y-h, w, h, stroke=0, fill=1)


def line(x, y, end, color=RULE):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(0.7)
    c.line(x, H-y, end, H-y)


def para(value, x, y, width, size=18, leading=26, color=INK):
    current = ''
    for word in value.split():
        new = (current + ' ' + word).strip()
        if current and pdfmetrics.stringWidth(new, 'Body', size) > width:
            text(current, x, y, size, color=color)
            current = word
            y += leading
        else:
            current = new
    if current:
        text(current, x, y, size, color=color)
    return y + leading


def slide(title=None, subtitle=None, dark=False):
    global page
    if page:
        c.showPage()
    page += 1
    rect(0, 0, W, H, INK if dark else PAPER)
    if title:
        text(title, 48, 63, 34, 'Bold')
    if subtitle:
        text(subtitle, 49, 91, 15, color=MUTED)
    line(48, 506, 912, '#436064' if dark else RULE)
    text('Сжатие изображений  /  CNNImageCodec', 48, 525, 10, color='#CBD9DA' if dark else MUTED)
    text(f'{page:02d} / 11', 912, 525, 10, color='#CBD9DA' if dark else MUTED, align='right')


def table(x, y, widths, headers, rows, row_h, size=15, bold_col=None):
    width = sum(widths)
    rect(x, y, width, row_h, TEAL)
    for i, (header, col_w) in enumerate(zip(headers, widths)):
        offset = x + sum(widths[:i])
        assert pdfmetrics.stringWidth(str(header), 'Bold', size) <= col_w - 8
        text(header, offset+col_w/2, y + row_h*.72, size, 'Bold', WHITE, 'center')
    for r, values in enumerate(rows):
        top = y + (r+1)*row_h
        if r % 2 == 0:
            rect(x, top, width, row_h, PALE)
        for i, (value, col_w) in enumerate(zip(values, widths)):
            offset = x + sum(widths[:i])
            font = 'Bold' if i == bold_col else 'Body'
            assert pdfmetrics.stringWidth(str(value), font, size) <= col_w - 8, (value, col_w)
            text(str(value), offset+col_w/2, top+row_h*.73, size, font,
                 TEAL if i == bold_col else INK, 'center')
    line(x, y+(len(rows)+1)*row_h, x+width)


def dec(value, places=3):
    return f'{float(value):.{places}f}'.replace('.', ',')


def pct(baseline, proposed):
    return 100 * (1 - float(proposed) / float(baseline))


slide(dark=True)
text('Лабораторная работа по теории информации', 50, 61, 18, color='#BFE2D9')
text('Контекстное кодирование', 48, 145, 52, 'Bold', WHITE)
text('признаков автокодера', 48, 204, 52, 'Bold', WHITE)
text('−15,42%', 48, 321, 68, 'Bold', '#8DE0C8')
text('BD-Rate относительно базового AE', 50, 359, 22, color=WHITE)
text('ФИО', 50, 438, 14, 'Bold', '#BFE2D9')
text(args.name, 122, 438, 18, color=WHITE)
text('Группа', 50, 467, 14, 'Bold', '#BFE2D9')
text(args.group, 122, 467, 18, color=WHITE)

slide('Модель вероятностей для range coder',
      'Цель работы: сократить битрейт минимум на 10% при одинаковом SSIM')
text('RGB 128×128  →  энкодер  →  латент 16×16×16  →  квантование  →  range coder',
     48, 141, 19, 'Bold', TEAL)
line(48, 162, 912)
text('Базовый контекст', 48, 204, 23, 'Bold')
para('Текущий канал и предыдущий символ в порядке обхода. Начальные счётчики одинаковы.',
     48, 242, 369)
para('В эксперименте переиспользуются нейросеть, квантование и арифметический кодер.',
     48, 335, 369, 16, 22, MUTED)
text('Контекст с выбором по данным', 466, 204, 23, 'Bold')
y = para('Для каждого канала выбираются пространственная подсказка и канал-подсказка. Начальные частоты оцениваются на train/.', 466, 242, 438)
para('После записи или чтения символа кодер и декодер увеличивают один и тот же счётчик.', 466, y+14, 438)
rect(48, 388, 864, 83, PALE)
text('Пример при b = 2', 65, 416, 18, 'Bold', TEAL)
text('Для канала 2 выбран канал 0 в той же точке и среднее соседей слева и сверху.',
     65, 447, 17)

slide('Подбор контекста и проверка',
      '204 изображения из train/ для подбора и 20 изображений из test/ для итоговой оценки')
para('Готовые энкодеры и декодеры для b = 1, 2, 3, 4 использованы без переобучения. Каждый пятый файл из train/ отложен для сравнения вариантов, остальные 163 использованы для подбора контекстов и частот.',
     48, 135, 864, 18, 24)
text('Экономия на 41 отложенном изображении относительно joint', 48, 223, 19, 'Bold')
modes = (('Канал-подсказка', 'learned-guide'), ('Пространственный сосед', 'learned-spatial'),
         ('Совместный выбор', 'learned-both'))
ablation = [[title, *[dec(pct(validation[b,'reference'], validation[b,mode]),2)+'%'
                      for b in range(1,5)]] for title, mode in modes]
table(48, 240, (296,142,142,142,142), ('Гипотеза', 'b = 1', 'b = 2', 'b = 3', 'b = 4'),
      ablation, 38, 16)
para('joint использует фиксированного соседа слева и предыдущий символ с частотами из train/. Совместный выбор победил для всех четырёх b.',
     48, 431, 860, 16, 22, MUTED)

slide('Средние результаты на 20 изображениях',
      'bpp = 8 × размер потока в байтах / (128 × 128). SSIM усредняется по RGB и по изображениям.')
means = [(r['b'], dec(r['baseline_bpp']), dec(r['improved_bpp']),
          dec(pct(r['baseline_bpp'], r['improved_bpp']), 1)+'%',
          dec(r['ae_ssim']), dec(r['jpeg_bpp']), dec(r['jpeg_ssim'])) for r in summary]
table(48, 124, (54,134,134,127,142,135,138),
      ('b', 'Баз. AE, bpp', 'Наш AE, bpp', 'Экономия', 'Оба AE, SSIM', 'JPEG, bpp', 'JPEG, SSIM'),
      means, 38, 15, bold_col=2)
text('−15,42% BD-Rate', 48, 368, 34, 'Bold', TEAL)
para('Расчёт по интерполяции PCHIP логарифма bpp при общем SSIM 0,6943-0,7920 в пределах заданного окна 0,2-0,9 bpp. Без экстраполяции.',
     48, 402, 850, 17, 23)
text('Порог в 10% выполнен. Все 80 пар потоков восстановили одинаковые признаки.',
     48, 476, 16, color=MUTED)

slide('Средний SSIM в зависимости от bpp')
c.drawImage(str(result/'rd-performance.png'), 40, H-90-396, width=618, height=396)
text('Точки для b = 1, 2, 3, 4', 688, 134, 19, 'Bold', TEAL)
y = para('При том же SSIM точки нашего AE лежат левее базового AE: для такого же качества нужно меньше бит.',
         688, 175, 219, 17, 23)
para('Для JPEG выбран ближайший доступный битрейт к базовому AE. Точки JPEG для b = 1 и b = 2 совпали.',
     688, y+21, 219, 17, 23, MUTED)

slide('Визуальное сравнение первых пяти изображений',
      'b = 2. Под восстановленными изображениями указаны bpp и SSIM.')
panel = Image.open(result/'comparison-b2.png')
cols = ((21,150), (190,319), (360,489), (530,659), (700,829))
panel_rows = ((15,144),(162,290),(309,437),(456,584))
b2 = [r for r in test if r['b']=='2'][:5]
for i,(xl,xr) in enumerate(cols):
    x = 188 + 145*i
    text(b2[i]['image'], x+42, 114, 12, 'Bold', MUTED, 'center')
    for r,(yt,yb) in enumerate(panel_rows):
        bounds = tuple(round(a*panel.width/850) for a in (xl,yt,xr,yb))
        tile = panel.crop(bounds)
        top = 122 + 92*r
        c.drawImage(ImageReader(tile), x+3, H-top-78, width=78, height=78)
        if r:
            rate, quality = (('baseline_bpp','ae_ssim'), ('improved_bpp','ae_ssim'), ('jpeg_bpp','jpeg_ssim'))[r-1]
            text(f'{dec(b2[i][rate])}  /  {dec(b2[i][quality])}', x+42, top+88,
                 11, 'Bold' if r==2 else 'Body', TEAL if r==2 else MUTED, 'center')
for r, label in enumerate(('Оригинал', 'Базовый AE', 'Наш AE', 'JPEG')):
    text(label, 48, 163+92*r, 17, 'Bold', TEAL if r==2 else INK)

slide('Воспроизведение эксперимента', 'Выполнить из корня репозитория на Linux с Python 3.11')
text('Репозиторий', 48, 129, 15, 'Bold', TEAL)
text(args.repo, 48, 156, 19)
if args.repo.startswith(('https://', 'http://')):
    c.linkURL(args.repo, (48, H-160, 910, H-137), relative=0, thickness=0)
rect(48, 184, 864, 150, INK)
for i, command in enumerate((
        'python3.11 -m venv .venv',
        '.venv/bin/python -m pip install -r requirements-benchmark.txt',
        '.venv/bin/python benchmark_context_map.py --output results/context-map')):
    text(command, 64, 213+32*i, 15, 'Mono', WHITE)
para('В results/context-map/ сохраняются таблицы, параметры контекстов и графики. Исходный range coder переиспользован.',
     48, 373, 865, 18, 25)
para('Размеры AE приведены для арифметических потоков. Масштаб деквантования и общие таблицы частот не входят в размер потока отдельного изображения.',
     48, 445, 865, 16, 22, MUTED)

for bits in ('1','2','3','4'):
    slide(f'Результаты по изображениям · b = {bits}',
          '20 файлов из test/. SSIM общий для базового и улучшенного AE.')
    rows = [(r['image'], dec(r['baseline_bpp']), dec(r['improved_bpp']),
             dec(pct(r['baseline_bpp'],r['improved_bpp']),1)+'%', dec(r['ae_ssim']),
             dec(r['jpeg_bpp']), dec(r['jpeg_ssim'])) for r in test if r['b']==bits]
    table(48, 108, (102,129,128,118,129,129,129),
          ('Файл', 'Баз. AE, bpp', 'Наш AE, bpp', 'Экономия', 'Оба AE, SSIM', 'JPEG, bpp', 'JPEG, SSIM'),
          rows, 18.5, 13, bold_col=2)

assert page == 11
c.save()
print(output)
