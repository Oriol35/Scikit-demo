# Guió de la demo — part de classificació i acció (≈ 6–7 min)

**Abans de començar**
- Obrir el dashboard (en línia o amb `demo.bat`) i anar a la pestanya **Classificació**.
- La primera vegada que s'obre la pestanya, els models tarden uns segons a entrenar-se.
- Deixar **Regressió logística**, **Pesos de classe** i **llindar 0,50** (valors per defecte).

---

## 1. Plantejament (≈ 30 s)

> «La supervivència ens deia **quan** marxa un client. Ara ens preguntem **si** marxarà, sí o no.
> Entrenem quatre models amb el 80 % dels clients i els posem a prova amb el 20 % que no han vist
> mai: els mateixos clients de prova que el model de Cox.»

- Assenyalar el requadre blau: explica què és el model triat, el seu punt fort i el seu punt feble.

## 2. Què surt d'un model (≈ 1 min)

> «Per a cada client, el model dona una **probabilitat de baixa**. Amb un **llindar** la convertim
> en una decisió: en risc o no.»

- Llegir la frase de sota les mètriques: dels **374** clients de prova que van marxar, el gradient
  boosting en detecta **303** (recall 81 %).
- Matriu de confusió: blau = encerts, taronja = errors.
- **Moure el llindar a 0,30**: el recall puja (≈ 91 %) però les falses alarmes també.

> «No hi ha un llindar correcte: depèn de quant ens costa perdre un client i quant ens costa fer
> una oferta innecessària.»

- Tornar el llindar a 0,50.

## 3. Desequilibri de classes (≈ 30 s)

- Canviar **Tractament** a **Cap**: el recall cau al ≈ 50–58 %.

> «Només marxa el 27 % dels clients. Sense tractament, el model aprèn a dir "no marxa". Amb pesos
> de classe, les baixes compten més i en detectem el 80 %. Fixeu-vos que l'exactitud puja quan el
> model és pitjor: per això no la fem servir.»

- Tornar a **Pesos de classe**.

## 4. Per què decideix així (≈ 1 min 30 s)

- **Regressió logística** → odds ratios:
  - Contracte de 2 anys: **OR 0,24** (redueix les odds un 76 %); fibra òptica: **OR 2,9**.
- **Arbre de decisió** → el dibuix de l'arbre:

> «La primera pregunta que fa és "contracte de 2 anys?". El camí de més risc és contracte mensual
> + fibra òptica + menys de 16 mesos: el 70 % d'aquests clients van marxar.»

- **Gradient boosting** → importància de les variables: antiguitat, contracte i internet.

> «Quatre models molt diferents coincideixen en les mateixes tres variables.»

## 5. Comparativa i campanya (≈ 1 min)

- Taula comparativa: AUC entre **0,815 i 0,839**. Les diferències són petites.
- Slider de campanya al **10 %**:

> «Si l'empresa només pot trucar a 141 clients, el model li diu a quins: en troba 102 dels 374 que
> marxaran, gairebé el triple que triant-los a l'atzar.»

## 6. Simulador (≈ 30 s)

- Client per defecte (mensual, fibra, 3 mesos): ≈ 85 % de probabilitat de baixa en tots els models.
- Canviar el contracte a **2 anys** o l'antiguitat a **48 mesos** i veure com baixa.
- Obrir «Camí que segueix aquest client a l'arbre de decisió».

> «El classificador ens diu **a qui** trucar; la supervivència ens diu **quan**.»

## 7. Perfils i campanyes (≈ 1 min 30 s)

- Pestanya **Perfils i campanyes**. Mapa de calor:

> «El risc és gairebé tot del contracte mensual: als primers 6 mesos marxa el 55 % dels clients;
> amb contracte d'1 any, al voltant del 10 %.»

- Recórrer els quatre perfils, cadascun amb la seva campanya: benvinguda, pas a anual, suport i
  seguretat, domiciliació.
- Simulador (20 % accepten, 10 % de descompte):

> «En l'escenari optimista la campanya d'anual guanya ≈ 15.500 €; en el prudent en perd ≈ 3.200 €.
> Només surt a compte si l'efecte real és com a mínim el 59 % del màxim. Per això proposem un pilot
> amb grup de control: és la idea de l'uplift modeling.»

- (Opcional) Pujar el descompte al 30 %: la campanya ja no surt a compte ni en l'escenari optimista.

---

## Pla B (si el dashboard no arrenca)

Gràfics exportats a `figures/` (HTML interactiu i PNG):

| Bloc | Fitxer |
|---|---|
| Matriu de confusió (gradient boosting) | `cl_confusio.html` |
| Corbes ROC dels 4 models | `cl_roc.html` |
| Odds ratios (regressió logística) | `cl_odds_ratios.html` |
| Arbre de decisió | `cl_arbre.html` |
| Importància de les variables | `cl_importancia.html` |
| Corba de guanys (campanya) | `cl_guanys.html` |
| Mapa de calor contracte × antiguitat | `pf_mapa_calor.html` |
| Perfils de risc vs. referència | `pf_perfils.html` |
| Simulador de la campanya d'anual | `pf_campanya.html` |

---

## Xifres clau per a l'informe (Word)

Conjunt de prova: 1.407 clients, 374 baixes. Llindar 0,5. Tractament: pesos de classe.

| Resultat | Valor |
|---|---|
| AUC | Logística 0,833 · Arbre 0,815 · Random Forest 0,836 · Gradient boosting 0,839 |
| Recall sense tractament / amb pesos / amb SMOTE | 48–58 % / 78–82 % / 74–79 % |
| Gradient boosting (llindar 0,5) | recall 81 % · precision 50 % · F1 0,62 · exactitud 73 % (VP 303, FP 303, FN 71, VN 730) |
| Efecte del llindar (gradient boosting) | 0,3: recall 91 %, precision 42 % · 0,7: recall 58 %, precision 63 % |
| Odds ratios destacats (logística) | 2 anys 0,24 · 1 any 0,46 · fibra 2,92 · sense internet 0,38 · xec electrònic 1,36 · seguretat 0,71 · suport 0,74 · antiguitat 0,970/mes (≈ 0,69/any) |
| Variables més importants (permutació) | Antiguitat, tipus de contracte i servei d'internet en els quatre models |
| Fulla de més risc de l'arbre | Mensual + fibra + antiguitat ≤ 15,5 mesos: 824 clients, 70 % de baixes |
| Campanya (gradient boosting) | 10 % contactat → 102 baixes (27 %), lift 2,7 · 20 % → 186 baixes (50 %), lift 2,5 |
| Comparació amb Cox (mateixos clients de prova) | correlació de rangs 0,89 · coincidència del 20 % de més risc 67 % · AUC del risc de Cox 0,79 |
| Limitacions | dataset transversal (sense finestres d'observació/predicció); `tenure` com a variable afavoreix el classificador; llindar sense costos reals; sense p-valors a la logística de scikit-learn |
| Taxa de baixa contracte × antiguitat | Mensual 55 % (1–6 m) · 42 % · 38 % · 33 % · 26 % (49–72 m) · 1 any 8–13 % · 2 anys 0–3 % |
| Perfils (taxa vs. referència · actius · baixes esperades 12 m) | Nous mensuals 55 % vs 36 % · 633 · ≈ 139 — Mensuals consolidats 34 % vs 12 % · 1.303 · ≈ 338 — Fibra sense suport ni seguretat 61 % vs 28 % · 599 · ≈ 209 — Xec electrònic 54 % vs 33 % · 856 · ≈ 264 |
| Campanya d'anual (20 % accepten, 10 % descompte) | risc 12 m 26 % · reducció màxima 66 % · optimista ≈ 45 baixes evitades, +15.500 € · prudent ≈ 22, −3.200 € · efecte mínim rendible 59 % |
