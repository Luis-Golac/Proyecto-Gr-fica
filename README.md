# CHAMO: Pipeline Híbrido para Generación de Avatares 3D desde una Imagen

## Descripción

El desarrollo eficiente y automático de avatares 3D animables desde imágenes individuales es un desafío crucial en aplicaciones emergentes como la realidad aumentada (AR) y virtual (VR). Este informe detalla el proyecto **CHAMO**, que propone un pipeline híbrido combinando **Gaussian Splatting**, **modelos de difusión avanzados** y técnicas modernas de **aprendizaje automático**.

CHAMO facilita la generación de modelos tridimensionales detallados y animables a partir de una sola imagen, asegurando la reproducibilidad y accesibilidad gracias al uso exclusivo de herramientas de código abierto (open-source). 

## Objetivos del Proyecto

-   #### Generar un avatar 3D de cuerpo completo desde una sola imagen.
    El principal objetivo del proyecto CHAMO es generar un avatar tridimensional integral que represente con fidelidad el cuerpo completo de un individuo utilizando únicamente una imagen de entrada. Este desafío implica extraer de la imagen original información precisa sobre geometría, postura, detalles faciales, texturas y características de vestimenta para construir un modelo 3D animable y robusto.

-   #### Lograr calidad visual comparable a escaneos multivista.
    Otro objetivo fundamental es lograr una calidad visual comparable a la obtenida mediante métodos tradicionales de escaneo multivista, que generalmente requieren múltiples cámaras y configuraciones controladas. CHAMO busca igualar o superar estos estándares con tecnologías avanzadas como Gaussian Splatting y Hunyuan3D-2 para capturar detalles finos y texturas realistas.

-   #### Inferencia en tiempo sub-minuto en una GPU de consumo.
    La eficiencia del proceso es clave, por lo que el pipeline está diseñado para realizar inferencias en tiempos inferiores al minuto utilizando GPUs estándar disponibles en el mercado, específicamente modelos como la NVIDIA RTX 4090. Esta característica garantiza que la solución sea viable en aplicaciones prácticas y no solo en ambientes académicos o experimentales.

-   #### Exportar modelos en formato malla (OBJ/PLY) y representaciones Gaussianas 3D.
    El pipeline ofrece exportación en formatos universales ampliamente utilizados en la industria y la academia, incluyendo `OBJ`, `PLY` y representaciones Gaussianas. Esta flexibilidad permite una integración sencilla con plataformas populares de desarrollo de gráficos y animación 3D como Blender, Unity o Unreal Engine.

-   #### Crear un flujo modular y reproducible con software libre.
    Finalmente, el proyecto busca democratizar el acceso a tecnologías avanzadas mediante una estructura modular bien documentada y completamente reproducible usando software de código abierto. Esto facilita la adopción y adaptación del pipeline para diversas aplicaciones y usuarios.

## Trabajos Relacionados y Fundamentos

El proyecto CHAMO integra y se basa en técnicas avanzadas previamente desarrolladas en el campo de la computación gráfica y la inteligencia artificial:

-   **Reconstrucción 3D desde una imagen:** Se inspira en técnicas como PIFu, ICON, Magic123 y DreamHuman.
-   **Gaussian Splatting:** Utiliza como referencia clave el trabajo de 3DGS (Inria) y NeuS-Splats.
-   **Generación 3D basada en difusión:** Incorpora modelos de vanguardia como Hunyuan3D-2, Zero123++ y DreamGaussian.
-   **Reconstrucción SfM:** Se apoya en las técnicas probadas de COLMAP para la estimación de poses de cámara.

## Módulos y Metodología

El pipeline está estructurado en una serie de módulos secuenciales que procesan los datos desde la captura hasta el modelo 3D final.

### 1. Captura y Pre-procesado

La captura inicial se realiza grabando un vídeo en resolución 4K a 30 fps con un smartphone. Se controla cuidadosamente la iluminación para que sea difusa y el fondo sea neutro, minimizando así complicaciones en etapas posteriores. Luego, se utiliza FFmpeg para extraer aproximadamente 2000 fotogramas y se estiman las poses iniciales de la cámara con COLMAP.

**Pasos detallados:**
1.  **Grabar vídeo** con un smartphone en **4K a 30 fps**, asegurando un fondo neutro y luz difusa.
2.  **Extraer frames** distribuidos uniformemente con FFmpeg. El siguiente comando extrae 10 frames por segundo:
    ```bash
    ffmpeg -i video.mp4 -vf fps=10 frames/frame_%04d.png
    ```
3.  **Ejecutar SfM** (Structure-from-Motion) con **COLMAP** para obtener las poses de cámara iniciales:
    ```bash
    colmap automatic_reconstructor \
     --image_path frames \
     --workspace_path sfm_output \
     --dense
    ```

### 2. LHM: Learning Human Gaussians

Utilizando el modelo avanzado LHM, se generan representaciones Gaussianas en 3D que incluyen posición, covarianzas, colores y opacidad, formando una base robusta para el avatar. LHM emplea un *transformer* multimodal y técnicas de atención global para fusionar efectivamente la información geométrica y visual extraída de la imagen original.

**Pasos detallados:**
1.  **Clonar el repositorio** de referencia `aigc3d/LHM` y preparar el entorno (Python 3.10, CUDA >= 11.8).
2.  **Utilizar el script `inference.sh`** para generar una representación Gaussiana 3D de la persona en pose-T:
    ```bash
    bash inference.sh LHM-1B ./input_images ./motion_smplx_params
    ```
3.  **El resultado** es un archivo `.PLY` con la representación gaussiana, que contiene:
    -   Posición (x, y, z)
    -   Covarianza (sigma)
    -   Color (RGB)
    -   Opacidad ($\alpha$)

### 3. 3D Gaussian Splatting

La representación Gaussiana inicial se refina visualmente con herramientas como `gs-viewer` y `GauStudio`. Esto permite renderizados interactivos en tiempo real y la generación eficiente de secuencias visuales que pueden ser exportadas.

-   Representación eficiente para renderizado en tiempo real.
-   Visualización y refinado con **gs-viewer** o **GauStudio**.
-   Exportación como vídeo o secuencia de vistas sintéticas.

### 4. Hunyuan3D-2: Modelo de Difusión para Texturizado

Este módulo avanzado de difusión utiliza el sistema **Hunyuan3D-2** para generar mapas de textura altamente detallados y coherentes. Lo hace mediante una generación multivista condicionada por la geometría. El modelo aprovecha técnicas como la atención multi-tarea y la selección eficiente de vistas para optimizar el resultado.

**Pasos detallados:**
1.  **Clonar el repositorio** `Tencent-Hunyuan/Hunyuan3D-2`.
2.  **Ejecutar inferencia** usando una imagen en pose-T o varias vistas. Para una sola imagen:
    ```bash
    python singleview.py \
     --image_path input.png \
     --output mesh_output.glb \
     --texture_dir texture_output \
     --steps 5 --fp16
    ```
3.  Si se dispone de un vídeo de 360°, se puede usar el **modo multivista** para mayor calidad:
    ```bash
    python multiview_cli.py \
     --image_dir frames \
     --output modelo.glb \
     --texture_dir texturas \
     --steps 5 --octree_res 380 --paint_steps 50 --fp16
    ```

### 5. Inferencia y Post-procesado

El proceso finaliza con la generación y refinamiento de mapas de profundidad, la aplicación de ruido controlado, la generación de un campo de distancia firmado truncado (TSDF) y la conversión a malla mediante el algoritmo de Marching Cubes. El post-procesado ajusta la escala y elimina puntos discrepantes (*outliers*) utilizando la información de SfM.

1.  Se genera un mapa de profundidad + ruido gaussiano.
2.  Un *denoiser* genera un TSDF que se convierte en malla (Marching Cubes).
3.  El texturizado final se realiza en la etapa "Paint" del modelo.
4.  SfM ajusta la escala y limpia los *outliers*.

## Resultados

CHAMO genera avatares visualmente ricos y detallados, compatibles con formatos y plataformas estándar. La calidad visual iguala a la de técnicas más complejas y costosas, validando la efectividad del pipeline híbrido propuesto.

-   **Avatar 3D de alta calidad** generado desde una sola imagen.
-   **Tiempo de inferencia inferior a 1 minuto** en una GPU NVIDIA RTX 4090.
-   **Exportación en formatos** `PLY`, `OBJ`, `GLB`, con texturas en `PNG`.
-   **Compatible** con Blender, Unity y Unreal Engine.

#### Galería de Ejemplos


## Limitaciones y Desafíos

Se reconocen limitaciones específicas, incluyendo:

-   **Materiales complejos:** La ropa reflectante o transparente disminuye la calidad de la reconstrucción.
-   **Cabello y detalles finos:** El cabello fino es particularmente difícil de capturar con precisión en la pose-T.
-   **Requerimientos de hardware:** Se requiere una GPU con **24 GB de VRAM o más** para la inferencia de alta calidad. Pudimos optimizar el proceso para GPUs de 8 GB de VRAM, pero se obtuvieron resultados de menor calidad (modelos 3D con baja resolución, estilo *low-poly*).
-   **Costos de entrenamiento:** Los costos asociados al entrenamiento inicial de estos modelos en la nube son elevados.

## Conclusiones y Futuro

El proyecto CHAMO ha demostrado la eficacia de integrar técnicas modernas en un pipeline reproducible y modular para generar avatares 3D. El pipeline unifica con éxito lo mejor de varias metodologías de código abierto, y la representación gaussiana subyacente permite un renderizado muy eficiente.

Futuros trabajos deberían centrarse en abordar las limitaciones actuales, explorando técnicas avanzadas de auto-supervisión, el manejo de múltiples poses de forma simultánea y la mejora del tratamiento de materiales complejos y deformables.


## Gestión del Proyecto

La gestión efectiva del proyecto se ha basado en una documentación rigurosa y una organización clara en repositorios Git, lo que facilita la colaboración y la replicabilidad completa del proceso.

-   **Repositorios integrados** como submódulos de Git para un manejo modular.
-   **Versionado reproducible** con commits fijos para garantizar la consistencia.
-   **Documentación interna y scripts** de ejemplo en cada etapa del pipeline.

## Repositorio y Material Adicional

-   **Repositorio en GitHub:** [**https://github.com/Luis-Golac/Proyecto-Grafica**](https://github.com/Luis-Golac/Proyecto-Grafica)
-   **Evidencias Visuales (Resultados):** [**Google Drive Link**](https://drive.google.com/drive/folders/158Q5FyuQa-Iufpe3F8zaNGboUipd60qP)
-   La **página web** del proyecto se encuentra en el código adjunto en el entregable final.

## Referencias

-   Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Fields*, SIGGRAPH 2023.
-   Zhao et al. *Learning Human Gaussian Models from Single Images*, arXiv 2024.
-   Tencent ARC. *Hunyuan3D-2: Instant single-image 3D avatar generation*, 2024.
-   Schönberger & Frahm. *Structure-from-Motion Revisited*, CVPR 2016.

