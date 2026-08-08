"""Capa de consenso por pelea: junta todo lo que el repo sabe y se lo da a un LLM.

`ufc/intel/` mira a un peleador a la vez y solo archiva noticias con cita. Esto es lo
otro: una pelea a la vez, con el modelo, el mercado, las features de los dos, la
inteligencia ya recolectada y las picks humanas en un solo prompt, para que salga un
veredicto que ninguna de esas piezas por separado puede dar.

    python -m ufc.ia.consenso --dry-run    # el prompt, sin gastar un peso
    python -m ufc.ia.consenso              # la cartelera entera
    python -m ufc.ia.evaluar               # si esto sirvio de algo, medido
"""
