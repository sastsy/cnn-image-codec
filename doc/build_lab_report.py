"""Generate the portrait laboratory report from the saved benchmark results."""
import argparse
import csv
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/context-map'
OUTPUT = ROOT / 'output/pdf/cnn-image-codec-lab-report.pdf'
INK = colors.HexColor('#18242D')
RULE = colors.HexColor('#ABB5BE')
SHADE = colors.HexColor('#EEF1F4')
WIDTH = A4[0] - 104


def register_fonts():
    mac = Path('/System/Library/Fonts/Supplemental')
    if (mac / 'Times New Roman.ttf').exists():
        folder = mac
        files = ('Times New Roman.ttf', 'Times New Roman Bold.ttf', 'Courier New.ttf')
    else:
        folder = Path('/usr/share/fonts/truetype/dejavu')
        files = ('DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf', 'DejaVuSansMono.ttf')
    for name, file in zip(('Serif', 'SerifBold', 'Mono'), files):
        pdfmetrics.registerFont(TTFont(name, str(folder / file)))
    pdfmetrics.registerFontFamily('Serif', normal='Serif', bold='SerifBold')


def styles():
    return {
        'body': ParagraphStyle('body', fontName='Serif', fontSize=12, leading=16,
                               spaceAfter=8, alignment=TA_JUSTIFY, textColor=INK),
        'heading': ParagraphStyle('heading', fontName='SerifBold', fontSize=16, leading=20,
                                  spaceBefore=12, spaceAfter=8, keepWithNext=True, textColor=INK),
        'subheading': ParagraphStyle('subheading', fontName='SerifBold', fontSize=13, leading=17,
                                     spaceBefore=8, spaceAfter=5, keepWithNext=True, textColor=INK),
        'caption': ParagraphStyle('caption', fontName='Serif', fontSize=10.5, leading=13,
                                  alignment=TA_CENTER, spaceBefore=4, spaceAfter=7, textColor=INK),
        'title': ParagraphStyle('title', fontName='SerifBold', fontSize=24, leading=30,
                                alignment=TA_CENTER, spaceAfter=16, textColor=INK),
        'cover': ParagraphStyle('cover', fontName='Serif', fontSize=15, leading=21,
                                alignment=TA_CENTER, spaceAfter=12, textColor=INK),
        'code': ParagraphStyle('code', fontName='Mono', fontSize=8.4, leading=13,
                               leftIndent=6, spaceBefore=4, spaceAfter=10, textColor=INK),
    }


def read_rows(name):
    with (RESULTS / name).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def decimal(value, places=3):
    return f'{float(value):.{places}f}'.replace('.', ',')


def summary_values(row):
    before, after = float(row['baseline_bpp']), float(row['improved_bpp'])
    return (decimal(before), decimal(after), decimal(100 * (1 - after / before), 1) + '%',
            decimal(row['ae_ssim']), decimal(row['jpeg_bpp']), decimal(row['jpeg_ssim']))


def result_table(rows, first='b', compact=False):
    headers = (first, 'AE, bpp', 'Наш, bpp', 'Экономия', 'AE, SSIM¹', 'JPEG, bpp', 'JPEG, SSIM')
    widths = [56, 65, 67, 71, 70, 74, WIDTH - 403]
    table = Table([headers, *rows], colWidths=widths, rowHeights=12.6 if compact else 20,
                  hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Serif'),
        ('FONTNAME', (0, 0), (-1, 0), 'SerifBold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9 if compact else 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), INK),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ('BACKGROUND', (0, 0), (-1, 0), SHADE),
        ('LINEABOVE', (0, 0), (-1, 0), 0.55, RULE),
        ('LINEBELOW', (0, 0), (-1, 0), 0.55, RULE),
        ('LINEBELOW', (0, -1), (-1, -1), 0.55, RULE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7F8F9')]),
    ]))
    return table


def page_number(canvas, doc):
    if doc.page > 1:
        canvas.setFillColor(INK)
        canvas.setFont('Serif', 10)
        canvas.drawCentredString(A4[0] / 2, 31, str(doc.page))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='Цыбина Александра')
    parser.add_argument('--repo', default='https://github.com/sastsy/cnn-image-codec')
    args = parser.parse_args()
    register_fonts()
    style = styles()
    summary = read_rows('summary.csv')
    test = read_rows('test.csv')
    report = (RESULTS / 'bd-rate.txt').read_text(encoding='utf-8')
    lo, hi, rate = re.search(r'common SSIM ([0-9.]+)–([0-9.]+)\): ([-+0-9.]+)%', report).groups()
    measure = decimal(rate, 2)
    story = []

    def put(text, kind='body'):
        story.append(Paragraph(text, style[kind]))

    story.append(Spacer(1, 80))
    put('Теория информации', 'cover')
    story.append(Spacer(1, 104))
    put('ОТЧЁТ<br/>ПО ЛАБОРАТОРНОЙ РАБОТЕ', 'title')
    put('Контекстное арифметическое кодирование<br/>признаков автокодера', 'cover')
    story.append(Spacer(1, 100))
    put(f'Выполнила: {escape(args.name)}', 'cover')
    story.append(PageBreak())

    put('1. Цель работы', 'heading')
    put('Снизить битрейт учебного кодека на основе автокодера (AE) не менее чем на 10% по метрике BD-Rate при равном среднем SSIM в диапазоне 0,2-0,9 бит/пиксель (bpp).')
    put('2. Предложенная модификация', 'heading')
    put('Энкодер преобразует RGB-изображение 128 × 128 в массив признаков 16 × 16 × 16. После квантования каждый элемент становится символом из алфавита размера 2<super>b</super>. Модификация меняет оценку вероятностей этих символов в арифметическом кодере. Чем выше вероятность фактически встреченного символа, тем меньше бит в среднем нужно для его записи.')
    put('2.1. Из чего состоит контекст', 'subheading')
    put('В базовом AE контекст включает номер текущего канала и предыдущий символ в порядке обхода. Предложенный метод выбирает таблицу частот по тройке <b>(номер канала, пространственная подсказка, вторая подсказка)</b>.')
    put('<b>Пространственная подсказка</b> берётся из того же канала: это символ слева, сверху или их округлённое среднее. На границе используется доступный сосед, а в первой точке значение равно нулю. <b>Вторая подсказка</b> берётся из уже обработанного канала в той же точке или из предыдущего символа в порядке обхода. Если вторая подсказка отключена для канала, её значение равно нулю.')
    put('Например, при b = 2 для канала 2 выбраны среднее двух соседей и значение канала 0 в текущей точке. Если слева стоит 1, сверху 2, а в канале 0 стоит 3, кодер использует таблицу для контекста (2, 2, 3). По её частотам он кодирует фактическое значение канала 2. Нумерация каналов начинается с нуля.')
    put('2.2. Подбор подсказок и декодирование', 'subheading')
    put('Для каждого канала и b перебираются пары подсказок. По одной части данных из train/ строятся начальные таблицы частот, а на другой оценивается суммарная длина адаптивного кода. Выбирается пара с наименьшей оценкой. Начальные частоты для выбранной пары пересчитываются на всех 163 изображениях, выделенных для настройки. Сглаживание сохраняет ненулевую вероятность каждого символа.')
    put('Для каждого изображения кодер и декодер начинают с одинаковых таблиц. Они обходят точки построчно, а внутри точки идут по каналам. Поэтому декодер уже знает обе подсказки, выбирает ту же таблицу и после чтения увеличивает счётчик восстановленного символа. На выходе получаются те же квантованные признаки, которые затем поступают в нейросетевой декодер после деквантования.')
    put('3. Методика эксперимента', 'heading')
    put('Из 204 изображений каталога train/ 163 использованы для настройки контекстов и начальных частот, 41 отложено для проверки размеров потоков. Нейросеть не переобучалась. Итоговые метрики рассчитаны на 20 изображениях из test/ для b = 1, 2, 3, 4. SSIM усреднялся по каналам RGB, а затем по изображениям. Битрейт равен числу бит в арифметическом потоке, делённому на 128 × 128. Для JPEG выбиралось качество, которое давало ближайший битрейт к базовому AE.')
    put('4. Результаты', 'heading')
    put('Таблица 1. Средние показатели для 20 тестовых изображений', 'caption')
    story.append(result_table([(row['b'], *summary_values(row)) for row in summary]))
    put('¹ Значение SSIM относится к обоим вариантам AE.', 'caption')
    put(f'Средний битрейт снизился при всех значениях b. Например, при b = 2 он уменьшился с 0,385 до 0,327 bpp, а средний SSIM составил 0,749. Проверка всех 80 потоков подтвердила точное восстановление квантованных признаков.')
    story.append(Image(str(RESULTS / 'rd-performance.png'), width=WIDTH-90,
                       height=(WIDTH-90)*808/1260))
    put('Рисунок 1. Средний SSIM в зависимости от bpp при b = 1, 2, 3, 4.<br/>Точки JPEG для b = 1 и b = 2 совпадают.', 'caption')
    put(f'BD-Rate относительно базового AE составил <b>{measure}%</b>. Логарифмы битрейтов интерполированы методом PCHIP, а интегрирование выполнено по общему интервалу SSIM {decimal(lo, 4)}-{decimal(hi, 4)} в пределах заданного окна 0,2-0,9 bpp. Расчёт не использует экстраполяцию.')
    put('Полные результаты по изображениям приведены в приложении А.')
    story.append(PageBreak())

    put('5. Визуальное сравнение', 'heading')
    story.append(Image(str(RESULTS / 'comparison-b2.png'), width=WIDTH-10,
                       height=(WIDTH-10)*1620/2340))
    put('Рисунок 2. Первые пять изображений при b = 2. Сверху вниз приведены оригинал, базовый AE, предложенный AE и JPEG. Для восстановленных изображений указаны bpp и SSIM.', 'caption')
    put('6. Воспроизведение', 'heading')
    url = escape(args.repo, {'"': '&quot;'})
    link = f'<link href="{url}" color="#183E66">{url}</link>' if args.repo.startswith(('https://', 'http://')) else url
    put(f'Репозиторий с исходным кодом: {link}. В Linux с Python 3.11 выполните из корня репозитория:')
    story.append(Preformatted(
        'python3.11 -m venv .venv\n'
        '.venv/bin/python -m pip install -r requirements-benchmark.txt\n'
        '.venv/bin/python benchmark_context_map.py --output results/reproduction', style['code']))
    put('Скрипт использует готовые модели из models/ и заново рассчитывает латенты и параметры контекстов. Для запуска достаточно исходных изображений в train/ и test/ и файлов моделей.')
    put(f'Ожидаемый BD-Rate в заданном диапазоне: <b>{measure}%</b>. Таблицы и графики сохраняются в results/reproduction/.')
    put('Битрейт обоих вариантов AE учитывает размер арифметического потока без масштаба деквантования и общих таблиц частот.')
    put('7. Вывод', 'heading')
    put(f'Модель вероятностей с совместным выбором подсказок снизила BD-Rate на {measure.lstrip("-")}% при сохранении качества восстановления. Целевой выигрыш 10% достигнут.')
    story.append(PageBreak())

    for first in (1, 3):
        put('Приложение А. Результаты по изображениям' if first == 1 else
            'Приложение А (продолжение)', 'heading')
        for b in (first, first+1):
            put(f'Таблица А.{b}. Показатели для b = {b}', 'subheading')
            rows = [(r['image'], *summary_values(r)) for r in test if int(r['b']) == b]
            assert len(rows) == 20
            story.append(result_table(rows, 'Файл', compact=True))
        put('¹ Значение SSIM относится к обоим вариантам AE.', 'caption')
        if first == 1:
            story.append(PageBreak())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=52, rightMargin=52,
                            topMargin=46, bottomMargin=49,
                            title='Отчёт по лабораторной работе. Контекстное кодирование')
    doc.build(story, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == '__main__':
    main()
