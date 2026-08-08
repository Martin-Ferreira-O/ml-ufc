"""Capa de analisis por pelea: un LLM leyendo el expediente deportivo de los dos.

`ufc/intel/` mira a un peleador a la vez y solo archiva noticias con cita. Esto es lo
otro: una pelea a la vez, con el record de los dos, por que via gana y pierde cada uno,
sus ultimas peleas y contra quien, las estadisticas de golpeo y lucha, y la inteligencia
reciente — para que salga la lectura de un analista, no otro numero.

**Ciega al mercado a proposito.** No ve cuotas, ni precios, ni la prediccion del modelo,
ni lo que eligieron los predictores humanos. Un LLM al que le mostras el precio deja de
analizar la pelea y empieza a explicar el precio. El EV se calcula despues, en Python,
cruzando su probabilidad con la cuota real: es la unica forma de que su lectura se pueda
medir CONTRA el mercado en vez de ser un eco de el.

    python -m ufc.ia.consenso --dry-run    # el prompt, sin gastar un peso
    python -m ufc.ia.consenso              # la cartelera entera
    python -m ufc.ia.evaluar               # si esto sirvio de algo, medido
"""
