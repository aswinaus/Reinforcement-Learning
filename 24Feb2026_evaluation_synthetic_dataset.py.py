# Databricks notebook source
import json
import random

# Base texts for positive (tax-related) and negative (non-tax) samples
positive_texts = [
    "However, a participation is (still) deemed to be held as a portfolio investment if: • The assets of the participation (on a consolidated basis) generally for more than 50% consist of investments in companies of less than 5%.12 Or The function of the subsidiary (and its subsidiaries) for more than 50% comprises of (in)direct group financing activities or financing of group assets (including the disposal of assets to group companies).13 To determine the function of the subsidiary, the type of turnover, the activities and the assets and liabilities of the entity in which the participation is held should be analyzed.14 Based on the parliamentary history, the concept of financing activities must be interpreted broadly.  ",
    "Assets that are used for activities that consist for more than 50% of group financing or group leasing and licensing activities are deemed to be free portfolio investments, unless (1) these activities qualify as active financing/active group leasing and licensing, or (2) the assets that are used for the financing - and leasing activities have been financed exclusively or almost exclusively with third party debt.17 Shareholdings of less than 5% are by definition considered free portfolio investments.",
    "Belanghebbende is een in Nederland gevestigde vennootschap die behoort tot een internationaal opererend concern. Aan belanghebbende zijn voor de jaren 2008 tot en met 2016 (navorderings)aanslagen vennootschapsbelasting (vpb) opgelegd met aanzienlijke correcties op de door belanghebbende aangegeven belastbare bedragen. Bovendien heeft de inspecteur voor de jaren 2010 en 2012 tot en met 2016 vergrijpboeten opgelegd. Voor alle jaren is in geschil of de vergoedingen (verrekenprijzen) die diverse in het buitenland gevestigde concernvennootschappen voor leveringen en diensten aan belanghebbende in rekening hebben gebracht als zakelijk kunnen worden aangemerkt. Daarnaast is voor het jaar 2016 in geschil of de beëindiging van door een gevoegde dochtermaatschappij van belanghebbende geëxploiteerde licentierechten dient te worden aangemerkt als een onzakelijke onttrekking aan het vermogen van belanghebbende.",
    "Aveva, quindi, acclarato che la contribuente aveva acquistato i beni dalla capogruppo ad un prezzo lontano dal ‘valore normale’ di cui al citato art. 110, comma 7. La società impugnava, con tre distinti ricorsi, gli atti impositivi innanzi alla Commissione tributaria provinciale di Firenze, che, previa riunione, li accoglieva: annullava il rilievo relativo al evidenziando che già era stata adottata, per altra annualità, una pronuncia favorevole alla contribuente, sulla base della non applicabilità della metodologia del TNMM e che l’Ufficio aveva errato nella scelta dei ‘comparabili’; annullava il rilievo relativo all’IVA in"
]

negative_texts = [
    "One of the most urgent international climate issues today is the accelerating rise in global temperatures, driven primarily by human activity. This warming trend—commonly referred to as global warming—is the result of increased concentrations of greenhouse gases such as carbon dioxide, methane, and nitrous oxide in the atmosphere. These gases trap heat, causing the Earth’s surface and oceans to warm at unprecedented rates.",
    "In the United States, after the 2008 economic crisis, “deaths of despair” rose among the working age population. Suicide and substance-use related mortality accounted for many of these deaths, which have been explained by lost hope due to unemployment, rising inequality and declining community support (58)."
]

# Function to generate a long text (~1000 words) by repeating and trimming
def generate_long_text(base_text, target_words=1000):
    words = base_text.split()
    repeats = target_words // len(words) + 1
    long_text = ' '.join(words * repeats)
    return ' '.join(long_text.split()[:target_words])

# Generate dataset
num_positive = 70
num_negative = 30
dataset = []

for _ in range(num_positive):
    base = random.choice(positive_texts)
    long_text = generate_long_text(base, 1000)
    dataset.append({"doc_text": long_text, "true_label": 1})

for _ in range(num_negative):
    base = random.choice(negative_texts)
    long_text = generate_long_text(base, 1000)
    dataset.append({"doc_text": long_text, "true_label": 0})

# Shuffle dataset
random.shuffle(dataset)

# Save to JSON
with open("evaluation_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Generated {len(dataset)} samples.")
