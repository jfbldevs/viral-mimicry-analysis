# Instrucciones para Subir a GitHub

El repositorio Git local ya está creado e inicializado con todos los archivos. Ahora necesitas crear el repositorio en GitHub y hacer push.

---

## Paso 1: Crear Repositorio en GitHub

1. **Ve a GitHub:** https://github.com/new

2. **Configuración del repositorio:**
   - **Repository name:** `viral-mimicry-analysis` (o el nombre que prefieras)
   - **Description:**
     ```
     Data and reproducibility code for: Simpson's paradox unmasks opposing
     evolutionary pressures constraining viral mimicry of human immune pathways
     ```
   - **Visibility:**
     - ✅ **Public** (recomendado para manuscrito académico)
     - O ⚪ Private (si aún no quieres que sea público)

   - **NO marques ninguna opción:**
     - ❌ NO marques "Add a README file"
     - ❌ NO marques "Add .gitignore"
     - ❌ NO marques "Choose a license"

     (Ya están creados localmente)

3. **Click en "Create repository"**

---

## Paso 2: Conectar y Subir el Repositorio

GitHub te mostrará una página con instrucciones. Usa la opción:
**"...or push an existing repository from the command line"**

En tu terminal (desde el directorio `data_and_scripts`), ejecuta:

```bash
# Agregar el remote (reemplaza 'tu-usuario' con tu username de GitHub)
git remote add origin https://github.com/tu-usuario/viral-mimicry-analysis.git

# Subir el código
git push -u origin main
```

**IMPORTANTE:** Reemplaza `tu-usuario` con tu username real de GitHub.

**Ejemplo:**
```bash
git remote add origin https://github.com/jorgebeltran/viral-mimicry-analysis.git
git push -u origin main
```

---

## Paso 3: Verificar

1. Ve a tu repositorio en GitHub: `https://github.com/tu-usuario/viral-mimicry-analysis`
2. Deberías ver:
   - ✅ README.md con descripción del proyecto
   - ✅ 12 scripts Python
   - ✅ Carpeta `sequences/` con proteomas
   - ✅ `virus_to_human_top5_neighbors_final_similarity.csv`
   - ✅ LICENSE con MIT + CC-BY 4.0
   - ✅ requirements.txt

---

## Paso 4: Obtener el DOI (Recomendado para Manuscrito)

### Opción A: Zenodo (Recomendado) - Genera DOI permanente

1. Ve a https://zenodo.org/
2. Inicia sesión con tu cuenta de GitHub
3. Ve a tu perfil → "GitHub" → "Sync now"
4. Busca tu repositorio y activa la integración
5. Ve a GitHub → Releases → "Create a new release"
   - **Tag version:** v1.0.0
   - **Release title:** "Initial Release - Manuscript Submission"
   - **Description:**
     ```
     First release for manuscript submission to [journal name].

     Contains:
     - Analysis scripts for viral mimicry study
     - Virus-human similarity dataset (3,945 pairs)
     - Protein sequences (50 viruses + human)
     - Complete reproducibility code
     ```
6. Click "Publish release"
7. Zenodo automáticamente creará un DOI

**Resultado:** Obtendrás un DOI permanente como: `10.5281/zenodo.XXXXXXX`

---

### Opción B: GitHub Release (Sin DOI)

Si NO quieres Zenodo ahora, al menos crea un release:

1. En GitHub, ve a tu repositorio
2. Click en "Releases" (sidebar derecho)
3. "Create a new release"
4. Tag: `v1.0.0`
5. Title: "Initial Release - Manuscript Submission"
6. Publish

---

## Paso 5: Actualizar el Manuscrito

Una vez que tengas el repositorio público, actualiza la sección **Data Availability** del manuscrito:

### Si usaste Zenodo (CON DOI):

```
All data and analysis scripts are available at:
https://github.com/tu-usuario/viral-mimicry-analysis

Archived version with DOI: https://doi.org/10.5281/zenodo.XXXXXXX
```

### Si NO usaste Zenodo (sin DOI):

```
All data and analysis scripts are available at:
https://github.com/tu-usuario/viral-mimicry-analysis
```

---

## URLs que Necesitas para el Manuscrito

Después de completar los pasos, tendrás:

1. **GitHub repository URL:**
   `https://github.com/tu-usuario/viral-mimicry-analysis`

2. **DOI (si usaste Zenodo):**
   `https://doi.org/10.5281/zenodo.XXXXXXX`

3. **Release URL:**
   `https://github.com/tu-usuario/viral-mimicry-analysis/releases/tag/v1.0.0`

Usa la URL de GitHub en el texto, y el DOI en la sección de Data Availability.

---

## Contenido del Repositorio

El repositorio incluye:

```
viral-mimicry-analysis/
├── README.md                              # Descripción del proyecto
├── LICENSE                                # MIT (código) + CC-BY 4.0 (datos)
├── requirements.txt                       # Dependencias Python
├── .gitignore                            # Archivos ignorados
│
├── virus_to_human_top5_neighbors_final_similarity.csv  # Dataset principal (3,945 pares)
│
├── sequences/                            # Secuencias proteicas (13 MB)
│   ├── human/
│   │   └── human_proteome.fasta         # Homo sapiens (UP000005640)
│   └── viruses/                         # 50 virus, 22 familias ICTV
│       ├── Herpesviridae/
│       ├── Retroviridae/
│       ├── Coronaviridae/
│       └── [otras familias]/
│
└── Scripts de análisis (12 archivos):
    ├── script_01_exploratory_analysis.py
    ├── script_02_distributions.py
    ├── script_03_normality_tests.py
    ├── script_04_global_correlation.py
    ├── script_05_stratified_correlation.py
    ├── script_06_benchmarking_internal.py
    ├── script_07_permutation_test.py
    ├── script_08_predictive_analysis.py
    ├── script_09_bar_visualization.py
    ├── script_11_normalized_family_comparison.py
    ├── script_KEGG_pathway_enrichment.py
    └── script_NN_baseline_final.py
```

**Tamaño total:** ~14 MB (perfecto para GitHub)

---

## Troubleshooting

### Error: "remote origin already exists"

```bash
git remote rm origin
git remote add origin https://github.com/tu-usuario/viral-mimicry-analysis.git
```

### Error: "Permission denied" al hacer push

Necesitas autenticarte. Opciones:

**Opción 1: Personal Access Token (recomendado)**
1. Ve a GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Marca "repo" scope
4. Copia el token
5. Al hacer push, usa el token como password

**Opción 2: SSH**
```bash
git remote set-url origin git@github.com:tu-usuario/viral-mimicry-analysis.git
```

### Hacer cambios futuros

```bash
# Hacer modificaciones
git add archivo_modificado.py
git commit -m "Descripción del cambio"
git push
```

---

## Checklist Final

Antes de submitir el manuscrito, verifica:

- [ ] Repositorio público en GitHub
- [ ] README visible y completo
- [ ] Todos los archivos subidos correctamente (69 archivos)
- [ ] Release v1.0.0 creado
- [ ] DOI de Zenodo obtenido (opcional pero recomendado)
- [ ] URL actualizada en la sección Data Availability del manuscrito
- [ ] License visible en GitHub (MIT + CC-BY 4.0)

---

## Comandos Resumen

```bash
# 1. Agregar remote (reemplaza tu-usuario)
git remote add origin https://github.com/tu-usuario/viral-mimicry-analysis.git

# 2. Push
git push -u origin main

# 3. Verificar
git remote -v
```

---

**¡Listo!** Tu código y datos ahora están disponibles públicamente para reproducibilidad científica.
