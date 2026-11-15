# Guió de la demo — part de supervivència (≈ 6 min)

**Abans de començar**
- Obrir un terminal a `CasPràctic/` i executar:
  `.venv\Scripts\python.exe -m streamlit run app/dashboard.py`
- Esperar que carregui (≈ 10 s la primera vegada) i **plegar la barra lateral** (fletxa «<») per
  guanyar amplada al projector.
- Tenir oberta la carpeta `figures/` com a pla B (vegeu al final).

---

## 1. Resum (≈ 30 s)

> «Treballem amb 7.032 clients. El 26,6 % ha marxat; el 73 % restant encara és client: són
> observacions **censurades**. Un classificador els tractaria com "no marxarà"; l'anàlisi de
> supervivència només diu "fins ara no ha marxat".»

- Assenyalar la corba global: **84 %** dels clients continuen al cap de 12 mesos; la mediana
  no s'assoleix.

## 2. Kaplan-Meier (≈ 1 min)

- Selector **Tipus de contracte** (ja és el primer).

> «La diferència és enorme: amb contracte mensual la meitat dels clients han marxat als
> **35 mesos**; amb contractes d'1 o 2 anys la mediana ni tan sols s'assoleix. El test log-rank
> ho confirma (χ² ≈ 2.350, p < 0,001).»

- Canviar a **Servei d'internet**: la fibra òptica marxa més.
- (Opcional) Treure l'interval de confiança per llegir-ho millor.

## 3. Model de Cox (≈ 1 min 30 s)

> «Kaplan-Meier mira una variable cada vegada. Cox les ajusta totes alhora i ens dona
> *hazard ratios*.»

- Llegir tres punts del gràfic:
  - **Contracte 2 anys: HR ≈ 0,07** → el risc instantani de baixa és 14 vegades menor que el mensual.
  - **Fibra òptica: HR ≈ 2,2** → més del doble de risc que DSL.
  - **Suport tècnic / seguretat en línia: HR ≈ 0,6–0,7** → serveis que retenen.
- Assenyalar el **C-index de test = 0,84**: «ordena correctament per risc el 84 % de parells de
  clients; és l'equivalent de l'AUC per a dades censurades».
- Obrir l'expander del **supòsit de riscos proporcionals** i activar **Estratificat per contracte**:

> «El model té límits: diverses variables violen el supòsit de riscos proporcionals, sobretot la
> quota mensual. Estratificant pel contracte corregim la violació del contracte i els altres efectes
> es mantenen, però variables com la quota tenen efectes que canvien amb el temps: és una
> limitació que documentem.»

## 4. Risc temporal (≈ 1 min)

> «Quan marxen? El risc té forma de U.»

- **Mes 1: 5,4 %** de baixes (gairebé **10 %** en contractes mensuals), davant d'un **0,5 %** mensual
  entre els mesos 6 i 60.
- Repunt al final (mesos 66–72), amb prudència perquè hi ha menys clients.
- Seleccionar els tres contractes al filtre per mostrar que el pic inicial és dels mensuals.

> «Conclusió operativa: la finestra crítica són els **primers 3–5 mesos**. No hem trobat pics
> a les renovacions dels mesos 12 o 24.»

## 5. Simulador de client — el moment "en directe" (≈ 1 min 30 s)

Els perfils inicials ja estan preparats:

| | Perfil A (inicial) | Perfil B (inicial) |
|---|---|---|
| Contracte | Mensual | 2 anys |
| Internet | Fibra òptica | DSL |
| Pagament | Xec electrònic | Targeta automàtica |
| Continua als 12 mesos | **≈ 50 %** | **≈ 98 %** |
| CLV esperat 24 m | ≈ 1.054 € | ≈ 1.415 € |

Accions en directe sobre el **perfil A**:
1. Marcar **Suport tècnic** i **Seguretat en línia** → S(12) puja a **≈ 76 %** i el CLV a **≈ 1.567 €**.
2. Desmarcar-los i canviar el contracte a **1 any** → S(12) **≈ 86 %**, CLV **≈ 1.750 €**.

> «Això és el que justifica una campanya: oferir suport tècnic o un contracte anual a un client
> com l'A augmenta el seu valor esperat a dos anys entre un 50 % i un 65 %.»

## 6. Risc × valor (≈ 15 s)

- Obrir **Risc × valor**:

> «Cada punt és un client actiu. A la dreta i a dalt hi ha els que tenen alt risc i alta quota:
> són els primers a retenir. Aquesta és la connexió entre el model i l'acció: el risc de Cox diu
> qui és probable que marxi i quan, i la quota i el CLV esperat diuen quant val retenir-lo.»

**Per a preguntes (opcional):** activar «Incloure també els clients que ja han marxat». Els punts
taronja són les baixes; la taula afegeix les columnes «(tots)».

> «Si barregem els que ja han marxat, la supervivència prevista gairebé no canvia, però el CLV
> s'infla fins a +81 € per client al segment Alt, perquè pagaven quotes més altes. Per això les
> decisions es prenen només amb els clients actius, que són les observacions censurades.»

---

## Pla B (si el dashboard no arrenca)

Tots els gràfics estan exportats a `figures/` en **HTML interactiu** (obrir amb el navegador,
amb zoom i *tooltips*) i en **PNG** per a l'informe:

| Bloc | Fitxer |
|---|---|
| Corba global | `km_global.html` |
| KM per contracte | `km_contracte.html` |
| Hazard ratios | `cox_forest.html`, `cox_forest_estratificat.html` |
| Risc temporal | `hazard_mensual.html`, `hazard_contracte.html` |
| Risc × valor | `risc_valor.html` |

---

## Xifres clau per a l'informe (Word)

| Resultat | Valor |
|---|---|
| Clients / baixes / censura | 7.032 / 1.869 (26,6 %) / 73,4 % |
| S(6), S(12), S(24) global | 88,5 % · 84,3 % · 78,9 % |
| Mediana per contracte | Mensual 35 mesos · 1 any i 2 anys: no assolida |
| S(12) per contracte | Mensual 70,3 % · 1 any 99,1 % · 2 anys 100 % |
| Log-rank (contracte) | χ² ≈ 2.353, p < 0,001 |
| Variables més discriminants (log-rank) | Contracte > seguretat en línia > suport tècnic > mètode de pagament > internet |
| HR destacats (Cox, train) | 2 anys 0,07 · 1 any 0,21 · fibra 2,2 · sense internet 0,34 · pagament automàtic ≈ 0,52 · seguretat 0,57 · suport 0,69 |
| C-index | train 0,859 · test 0,839 · estratificat (test) 0,798 |
| Supòsit PH | violat per quota mensual, contracte 1 any, pagament automàtic, suport tècnic, parella, seguretat |
| Weibull vs exponencial (AIC) | 21.157 vs 21.697 → el risc no és constant |
| Risc temporal | mes 1: 5,4 % (mensuals 9,8 %); mesos 6–60: 0,5 % mensual; mesos crítics 1, 2, 3, 70, 72 |
| Segments de risc (Cox, clients actius) | Alt: S(12) 69 %, CLV 24m ≈ 1.068 € · Mitjà: 85 %, ≈ 1.164 € · Baix: 95 %, ≈ 1.375 € |
| Limitacions | dades transversals (data de tall desconeguda); supòsit PH violat per diverses variables (l'estratificació només corregeix el contracte); més enllà de 72 mesos el risc base s'extrapola linealment; CLV sense descompte i amb quota actual; RSF no executat (scikit-survival no disponible per a Python 3.14) |
