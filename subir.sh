#! /bin/sh
#+ Autor:	Ran#
#+ Creado:	24/07/2021 00:19:42
#+ Editado:	24/07/2021 00:32:50

case $1 in
    's')
        rm saidas.7z
        7z a saidas saidas/ -mhe=on -p"$(cat .contrasinal)"
        rm -rf saidas/
        gc $2
        7z x saidas.7z -p"$(cat .contrasinal)"
        ;;
    'd')
        7z x saidas.7z -p"$(cat .contrasinal)"
        ;;
    '?')
        echo '$1 → s/d subir ou descomprimir saidas.7z'
        echo '$2 → mensaxe de commit'
        ;;
esac
