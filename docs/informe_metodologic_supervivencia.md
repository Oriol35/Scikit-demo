# Informe metodològic: ús del dataset Telco Customer Churn per a supervivència

## 1. Objectiu i informació disponible

El dataset conté 7.043 clients, amb:

- `tenure`: antiguitat en mesos;
- `Churn = "Yes"`: el client ha abandonat;
- `Churn = "No"`: el client consta com a client actiu en el moment de l'observació.

La definició proposada és:

```text
duration = tenure
event = 1 si Churn == "Yes"
event = 0 si Churn == "No"
```

La inspecció exploratòria ha confirmat que hi ha 1.869 abandonaments i 5.174 observacions amb `Churn = "No"`. També ha confirmat que `tenure` és numèric, no negatiu i està comprès entre 0 i 72 mesos.

El problema principal no és tècnic sinó d'interpretació: el fitxer no proporciona explícitament dates d'alta, dates de baixa, data de tancament de l'estudi ni el mecanisme pel qual els clients van ser seleccionats.

## 2. És defensable aplicar Kaplan-Meier?

### Resposta curta

**Sí, aplicarem Kaplan-Meier i Cox amb una definició operativa explícita.** Assumirem que `tenure` és el temps observat, que `Churn = "Yes"` és l'esdeveniment i que `Churn = "No"` és una observació censurada a la seva antiguitat. La limitació principal es documentarà, però no impedeix continuar l'anàlisi.

L'estimador de Kaplan-Meier requereix, com a mínim:

1. un temps d'origen comú i ben definit;
2. un temps d'observació per a cada individu;
3. saber si l'observació acaba per esdeveniment o per censura;
4. observacions independents, o una metodologia que tingui en compte la dependència;
5. censura no informativa, o independent del temps fins a l'esdeveniment condicionada a la informació utilitzada.

En un estudi retrospectiu, no és imprescindible que les dates estiguin físicament en el CSV si `tenure` representa exactament el temps des de l'alta fins a la baixa o fins a una data de tall comuna. En aquest cas, la definició podria aproximar una anàlisi de temps fins al primer churn:

- `Churn = "Yes"`: esdeveniment observat en el mes de churn;
- `Churn = "No"`: censura dreta en el mes d'antiguitat observat.

Tanmateix, el dataset disponible no permet verificar que aquest sigui realment el procés de generació de les dades. Per això, la corba es pot calcular, però l'estimand i la validesa externa no queden establerts.

## 3. Hipòtesis que s'estarien assumint

Amb la definició proposada, estaríem assumint simultàniament:

### 3.1. Interpretació correcta de `tenure`

`tenure` seria el temps exacte entre l'inici de la relació contractual i:

- el churn, per als clients amb `Churn = "Yes"`; o
- la data comuna de tall, per als clients amb `Churn = "No"`.

Si `tenure` és només una antiguitat registrada en el moment de crear el dataset, aquesta interpretació és plausible, però no queda demostrada sense documentació del procés de captura.

### 3.2. `Churn = "No"` equival a censura dreta

Estaríem assumint que aquests clients encara no havien abandonat en la data de tall i que, per tant, no són clients que simplement no van respondre, es van perdre del sistema o van ser exclosos.

### 3.3. Censura no informativa

La probabilitat de ser censurat hauria de ser independent del risc de churn, un cop considerades les covariables rellevants. En termes pràctics, els clients observats com a `No` haurien de tenir el mateix futur potencial que els clients comparables que continuen en seguiment.

És una hipòtesi forta en una fotografia transversal. Per exemple, si alguns tipus de contracte o perfils de risc deixen de ser observables abans, el tractament de tots els `No` com a censures pot esbiaixar la corba.

### 3.4. Origen i selecció comparables

Els clients haurien de formar part d'una població de risc definida i comparable des del moment en què comença el temps. Si el dataset barreja clients reclutats en moments diferents o només inclou els supervivents disponibles en una data, caldria considerar entrada tardana, truncament esquerre o biaix de selecció.

### 3.5. Un sol esdeveniment d'interès

El model tracta el primer churn com l'esdeveniment principal. No considera reactivacions, canvis de contracte, pauses, migracions, defuncions del compte per causes diferents ni riscos competitius.

## 4. Problemes de censura i biaixos possibles

### 4.1. Censura dreta amb data de tall desconeguda

Per als clients amb `Churn = "No"`, `tenure` només pot funcionar com a temps de censura si representa l'antiguitat fins a una data de tall comuna. Si les observacions corresponen a dates de tall diferents, els temps no són directament comparables.

La censura administrativa sol ser una situació raonable per a Kaplan-Meier, però ha d'estar definida. Amb el CSV actual no podem comprovar:

- quina és la data de tall;
- si tots els clients estaven sota observació fins a aquesta data;
- si els clients `No` van continuar sent seguits després de la fotografia;
- si es van excloure comptes amb informació incompleta.

### 4.2. Truncament esquerre i entrada tardana

En una mostra de prevalença, els clients que van abandonar abans de la data de selecció poden no aparèixer, mentre que els clients amb més antiguitat tenen més probabilitat d'estar presents. Això modifica el conjunt de risc observat.

El `tenure` observat no resol per si sol aquest problema: cal conèixer quan cada client va entrar en el mecanisme de mostreig i quins clients podien ser inclosos.

### 4.3. Biaix de selecció

La mostra pot no representar totes les altes inicials. Pot contenir una combinació de:

- clients actius en una data concreta;
- clients que ja han marxat però encara consten en el sistema;
- clients amb registres complets;
- clients seleccionats segons criteris comercials o de disponibilitat.

Si la inclusió depèn de la supervivència o de variables relacionades amb el churn, la corba estimada pot no representar una cohort d'altes.

### 4.4. Biaix de supervivència i biaix de durada

Les mostres transversals poden sobre-representar els episodis llargs perquè duren més temps i, per tant, tenen més probabilitat de ser observats en una fotografia. Aquest fenomen es coneix com a *length bias* o biaix de durada.

El biaix pot fer que la retenció sembli millor que en una cohort que es segueix des de l'alta. L'estimador Kaplan-Meier estàndard no corregeix automàticament aquest mecanisme de mostreig.

Cal distingir aquest problema del fet que el dataset sí que inclogui clients amb `Churn = "Yes"`: la presència d'aquests clients redueix la preocupació que la mostra contingui exclusivament supervivents, però no demostra que la selecció sigui una cohort d'incidència ni elimina el possible biaix de prevalença.

### 4.5. Variables mesurades en un moment posterior

Variables com el tipus de contracte, el mètode de pagament o alguns serveis poden haver canviat durant la relació amb l'empresa. Si s'utilitzen com si fossin covariables basals, es pot produir biaix temporal o una interpretació causal incorrecta.

## 5. Com s'haurien d'interpretar les corbes Kaplan-Meier?

Les corbes es poden interpretar com una estimació de la probabilitat de romandre sense churn al llarg dels mesos dins de la mostra i sota la definició adoptada. La censura indica que, per als clients amb `Churn = "No"`, no s'ha observat l'abandonament durant el temps disponible.

També cal evitar confondre:

- **supervivència:** probabilitat estimada de no haver experimentat churn fins a un temps determinat;
- **hazard:** risc instantani o condicional d'experimentar churn entre clients que encara no han abandonat;
- **incidència acumulada de churn:** proporció o probabilitat acumulada d'haver abandonat fins a un temps.

## 6. El model de Cox té les mateixes limitacions?

**Sí.** Cox utilitza la mateixa informació de temps i esdeveniment, de manera que hereta:

- l'ambigüitat de `tenure`;
- la classificació de `Churn = "No"` com a censura;
- la possible censura informativa;
- el biaix de selecció, supervivència i durada;
- la falta d'una població d'origen i d'un calendari de seguiment clar.

Addicionalment, Cox requereix hipòtesis pròpies:

1. **Riscos proporcionals:** la raó de riscos entre grups és aproximadament constant en el temps;
2. **Linealitat en el log-hazard** per a les covariables numèriques, o una especificació alternativa adequada;
3. **Absència de col·linearitat problemàtica** i suficient nombre d'esdeveniments;
4. **Covariables mesurades en un moment coherent**, idealment abans de l'esdeveniment i sense incorporar informació posterior;
5. **Independència de la censura**, igual que en Kaplan-Meier.

Per tant, el model de Cox es pot utilitzar amb la mateixa definició de temps i censura. Caldrà comprovar l'assumpció de riscos proporcionals i interpretar els hazard ratios com associacions observades dins del dataset.

## 7. Alternatives per obtenir una base adequada

### Alternativa recomanada: cohort longitudinal de clients

Caldria disposar, com a mínim, d'una taula per client amb:

| Camp | Funció |
|---|---|
| `customerID` | identificador estable |
| `start_date` | data d'alta o inici del contracte |
| `end_date` | data de churn, si existeix |
| `last_observed_date` | última data de seguiment |
| `churn_event` | 1 si hi ha churn, 0 si censurat |
| covariables basals | informació coneguda a l'inici |
| canvis de contracte/servei | covariables dependents del temps, si cal |

La data de finalització hauria de ser `end_date` per als abandonaments i `last_observed_date` o la data de tall per als censurats.

### Fonts o dissenys adequats

1. **Registres de CRM o facturació:** altes, baixes, reactivacions, canvis de contracte i data de l'última factura.
2. **Extractes mensuals de clients:** una fila per client i mes, amb estat actiu i serveis contractats.
3. **Cohort d'altes noves:** seguir tots els clients que entren durant un període definit fins al churn o la data de tall.
4. **Esdeveniments de servei:** taules d'incidències, cancel·lacions i renovacions amb marques temporals.
5. **Dades sintètiques o simulades amb dates conegudes:** útils per validar el codi, però no per treure conclusions empresarials reals.

Si no es pot obtenir temps longitudinal, una alternativa més honesta és tractar el problema com una classificació transversal de churn (`Churn` com a variable resposta), deixant clar que no estima supervivència ni hazard.

## 8. Decisió recomanada per al projecte

Es recomana **mantenir el dataset actual i continuar amb Kaplan-Meier, log-rank i Cox**. La codificació de supervivència serà:

```python
duration = tenure
event = (Churn == "Yes").astype(int)
```

La censura correspon als clients amb `Churn == "No"`.

Abans de calcular Kaplan-Meier, l'informe hauria d'incloure una declaració com aquesta:

> En aquest projecte, `tenure` es considera la durada observada. `Churn = "Yes"` es codifica com a esdeveniment i `Churn = "No"` com a censura dreta. Aquesta definició permet aplicar Kaplan-Meier, el test log-rank i el model de Cox. Com que el dataset és una fotografia i no inclou les dates originals, els resultats s'interpretaran com estimacions basades en la informació disponible.

Si més endavant es disposa d'una font amb dates d'alta i baixa, es podrà repetir l'anàlisi amb una base longitudinal més completa. Aquesta millora és recomanable, però no és necessària per completar la fase actual del projecte.

## 9. Fonts bibliogràfiques i documentació tècnica

- National Cancer Institute. [Survival analysis](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/survival-analysis).
- Kleinbaum, D. G. i Klein, M. *Survival Analysis: A Self-Learning Text*. Springer. Referència general sobre censura, Kaplan-Meier i Cox.
- Clark, T. G. et al. (2003). [Survival analysis part I: basic concepts and first analyses](https://pmc.ncbi.nlm.nih.gov/articles/PMC2394262/). *British Journal of Cancer*.
- Bewick, V., Cheek, L. i Ball, J. (2004). [Statistics review 12: survival analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC1065034/). *Critical Care*.
- [lifelines documentation](https://lifelines.readthedocs.io/en/latest/), documentació tècnica de Kaplan-Meier i Cox en Python.
- Statistical Society of Canada. [Recent advances in analysis of survival data from cross-sectional sampling](https://ssc.ca/en/recent-advances-analysis-survival-data-cross-sectional-sampling), sobre mostreig transversal, biaix de durada i estimació de supervivència.
- [Cross-sectional studies: strengths, limitations, and methodological considerations](https://pmc.ncbi.nlm.nih.gov/articles/PMC13665586/), discussió de les limitacions dels dissenys transversals.

