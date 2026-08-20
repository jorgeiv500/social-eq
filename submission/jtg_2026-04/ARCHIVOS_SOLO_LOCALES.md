# Archivos excluidos del repositorio público

`github.com/jorgeiv500/social-eq` es **público** y se declaró como *Research Data* en el
envío (p. 19 de la ficha de Editorial Manager), así que los revisores pueden abrirlo.
El artículo está en **revisión doble ciego**.

Por eso estos cuatro archivos existen en esta carpeta pero **no se versionan**:

- `01_manuscrito/00_editorial_manager_JTRG-D-26-00692.pdf` — ficha con nombres, orden de autores y correspondencia
- `01_manuscrito/03_title_page_author_details.pdf` — portada con filiaciones y CRediT
- `01_manuscrito/04_cover_letter_to_editor.pdf` — carta firmada por los tres autores
- `00_latex_fuente/data_additional/title_page_author_details.tex` — fuente de la portada

Lo demás sí se versiona: `main.tex`, `supplementary_appendix.tex`, las figuras, las tablas,
el código y los dos PDF anonimizados. Se verificó que ninguno menciona a los autores
ni sus filiaciones.

Cuando termine el proceso de revisión anónima se pueden incorporar quitando estas
cuatro líneas de `.gitignore`.
