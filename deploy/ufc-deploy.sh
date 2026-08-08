#!/usr/bin/env bash
# Trae la rama de GitHub y reinicia el bot si cambio el codigo. Lo corre ufc-deploy.timer.
#
# Es todo el CI/CD que este proyecto necesita: el repo es publico, asi que no hay
# credenciales que rotar, y un `git fetch` cada pocos minutos cuesta menos que mantener un
# runner o abrir un puerto para un webhook.
#
# El `reset --hard` solo corre cuando hay algo nuevo. Importa: entre despliegues el bot
# escribe `data/resultados.csv` y el de inteligencia appendea `data/cartelera_hist.csv`,
# los dos versionados. Reseteando siempre les pasarias el trapo cada cinco minutos; asi
# solo se pisan cuando efectivamente hay un commit nuevo, y el bot los reescribe desde
# ESPN en el tick siguiente.
set -euo pipefail

RAIZ="${UFC_RAIZ:-/opt/ml-ufc}"
RAMA="${TIPSTER_RAMA:-ufc-fight-predictor}"
SERVICIO="ufc-tipster.service"

cd "$RAIZ"
git fetch --quiet origin "$RAMA"

LOCAL=$(git rev-parse HEAD)
REMOTO=$(git rev-parse "origin/$RAMA")
if [ "$LOCAL" = "$REMOTO" ]; then
    exit 0
fi

echo "Desplegando ${LOCAL:0:7} -> ${REMOTO:0:7}"
# Que cambio, antes de movernos. Sirve para decidir si hay que reinstalar dependencias o
# recargar systemd, y queda en el journal como registro de que se desplego.
CAMBIOS=$(git diff --name-only "$LOCAL" "$REMOTO")
echo "$CAMBIOS" | sed 's/^/  /'

git reset --hard --quiet "origin/$RAMA"

if echo "$CAMBIOS" | grep -q '^requirements.txt$'; then
    echo "requirements.txt cambio: actualizando el venv"
    "$RAIZ/.venv/bin/pip" install --quiet --upgrade -r "$RAIZ/requirements.txt"
fi

if echo "$CAMBIOS" | grep -q '^deploy/.*\.\(service\|timer\)$'; then
    echo "unidades systemd cambiaron: reinstalando"
    # Las del repo son plantillas con REEMPLAZAR_USUARIO/GRUPO adentro. Copiarlas crudas
    # dejaria unidades que no arrancan, y encima justo despues de un deploy exitoso.
    for unidad in "$RAIZ"/deploy/*.service "$RAIZ"/deploy/*.timer; do
        sed -e "s/REEMPLAZAR_USUARIO/$(id -un)/" -e "s/REEMPLAZAR_GRUPO/$(id -gn)/" \
            "$unidad" | sudo tee "/etc/systemd/system/$(basename "$unidad")" >/dev/null
    done
    sudo systemctl daemon-reload
fi

# Solo el codigo justifica un reinicio. Un push que toca unicamente `data/` (que es el
# caso normal: picks nuevos antes de una cartelera) lo levanta el bot solo, que ya
# sincroniza esos CSV en cada vuelta.
if echo "$CAMBIOS" | grep -qE '\.py$'; then
    echo "codigo python cambio: reiniciando $SERVICIO"
    sudo systemctl restart "$SERVICIO"
fi

echo "Listo en $(git rev-parse --short HEAD)"
