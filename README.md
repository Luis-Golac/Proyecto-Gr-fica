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

![Video rotatorio](images/Hunyuan_DiT_example.png)

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

**LHM_DataGen y LHM_DataGen_2**

![Generación de Data - LHM](images/LHM-DataGen.png)

![Arquitecura - LHM](images/LHM-DataGen_2.png)

### 3. 3D Gaussian Splatting

La representación Gaussiana inicial se refina visualmente con herramientas como `gs-viewer` y `GauStudio`. Esto permite renderizados interactivos en tiempo real y la generación eficiente de secuencias visuales que pueden ser exportadas.

-   Representación eficiente para renderizado en tiempo real.
-   Visualización y refinado con **gs-viewer** o **GauStudio**.
-   Exportación como vídeo o secuencia de vistas sintéticas.

### 4. Hunyuan3D-2: Modelo de Difusión para Texturizado

Este módulo avanzado de difusión utiliza el sistema **Hunyuan3D-2** para generar mapas de textura altamente detallados y coherentes. Lo hace mediante una generación multivista condicionada por la geometría. El modelo aprovecha técnicas como la atención multi-tarea y la selección eficiente de vistas para optimizar el resultado.


![Video rotatorio](images/Hunyuan_mesh.png)


![Video rotatorio](images/Hunyuan_DiT.png)


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

![Video rotatorio](images/Hunyuan_DiT_example.png)

### 5. Inferencia y Post-procesado

El proceso finaliza con la generación y refinamiento de mapas de profundidad, la aplicación de ruido controlado, la generación de un campo de distancia firmado truncado (TSDF) y la conversión a malla mediante el algoritmo de Marching Cubes. El post-procesado ajusta la escala y elimina puntos discrepantes (*outliers*) utilizando la información de SfM.

1.  Se genera un mapa de profundidad + ruido gaussiano.
2.  Un *denoiser* genera un TSDF que se convierte en malla (Marching Cubes).
3.  El texturizado final se realiza en la etapa "Paint" del modelo.
4.  SfM ajusta la escala y limpia los *outliers*.



![Video rotatorio](images/Hunyuan_Paint.png)


![Video rotatorio](images/Hunyuan_Paint_example.png)

## Modificaciones Realizadas


### Código de Entrenamiento ([`src/train.py`](src/train.py))

Esta sección detalla la implementación del script de entrenamiento principal del repositorio.

#### 1. Carga de Datos: La Clase `VideoFrameDataset`

La preparación de los datos es manejada por la clase `VideoFrameDataset`, diseñada para procesar los datos de video y geometría.

-   **Función Principal:** Su objetivo es leer directorios que contienen los fotogramas de un video (`.jpg`) y un archivo `smplx.json` con los metadatos de la geometría.
-   **Muestreo (`__getitem__`)**: Para cada video, el método `__getitem__` selecciona aleatoriamente un conjunto de `n_source` imágenes de referencia y `n_target` imágenes de destino.
-   **Metadatos de Geometría**: Carga de forma crítica los parámetros del modelo paramétrico **SMPL-X** desde `smplx.json`. El código extrae `betas` (forma), `thetas` (pose), `expressions` y `camera`, que son tensores esenciales para condicionar el modelo.


#### 2. Arquitectura del Modelo: El Transformer `LHM`

La arquitectura principal, implementada en la clase `LHM`, es un **Transformer multimodal** que fusiona diferentes tipos de información.

-   **Bloques Fundamentales**: El modelo se construye a partir de módulos estándar de Transformer:
    -   `Attention`: Una implementación de atención multi-cabeza.
    -   `Block`: Un bloque de Transformer completo que combina atención, una MLP y normalización de capa (`LayerNorm`).
    -   `BHTransformer`: Una pila de `Block` que forma el codificador principal.
-   **Flujo de Datos en `LHM.forward()`**:
    1.  **Entradas**: El método `forward` recibe `body_pts` (puntos de la geometría), `image_tok` (tokens de la imagen de referencia) y `head_tok` (tokens de la cabeza).
    2.  **Codificación de Posición**: La función `fourier_emb` convierte las coordenadas 3D de `body_pts` en un embedding de alta frecuencia, permitiendo al modelo entender mejor la posición espacial.
    3.  **Fusión**: Los tres tipos de tokens se proyectan a una dimensión común y se **concatenan** en una única secuencia. Este es el paso clave de la fusión multimodal.
    4.  **Procesamiento**: La secuencia fusionada pasa a través del `BHTransformer`, donde la información se mezcla e integra.
    5.  **Decodificador (`GaussianDecoder`)**: Finalmente, los tokens de salida asociados a los puntos del cuerpo se procesan con una MLP para predecir los **16 parámetros** que definen cada una de las Gaussianas 3D del avatar.

#### 3. Función de Pérdida y Renderizado

La optimización del modelo se guía por una función de pérdida compuesta definida en el script.

-   **Pérdida Fotométrica**: La función `photometric_loss` calcula la diferencia (L1 + MSE) entre la imagen renderizada y la imagen objetivo real.
-   **Regularización Geométrica**: Las funciones `asap_loss` y `acap_loss` son cruciales. No operan sobre la imagen, sino directamente sobre los parámetros de las Gaussianas 3D para asegurar que la geometría sea coherente y no se degrade.
-   **Renderizador (`splat_render`)**: **Importante**: En el código del repositorio, `splat_render` es una **función placeholder** que retorna un tensor de ceros. Para que el entrenamiento funcione, esta debe ser reemplazada por un **renderizador diferencial de splatting** real.

#### 4. Sistema de Entrenamiento (`LHMSystem`) y Ejecución

El bucle de entrenamiento se abstrae utilizando la clase `LHMSystem` de PyTorch Lightning.

-   **Módulo Lightning (`LHMSystem`)**: Encapsula el modelo `LHM` y define:
    -   `training_step`: Contiene la lógica para un paso de entrenamiento. **Nota**: En el código actual, `body_pts`, `img_tok` y `head_tok` se inicializan con datos aleatorios (`torch.randn`). Esto indica que se necesita un codificador de imágenes (como CLIP) y una estrategia de muestreo de puntos para una ejecución real.
    -   `configure_optimizers`: Configura el optimizador `AdamW` y un planificador de tasa de aprendizaje `CosineAnnealingLR`.

### Estructura de Datos: Octree ([`src/octree.py`](src/octree.py))

Para optimizar las operaciones espaciales, como la búsqueda de vecinos cercanos en nubes de puntos, el proyecto utiliza una implementación de un **Octree**. Esta es una estructura de datos de árbol en la que cada nodo interno tiene exactamente ocho hijos. Es especialmente eficiente para particionar un espacio tridimensional de forma recursiva.

#### 1. La Clase `NodoOctree`

Esta clase, definida como un `dataclass`, representa un único nodo (un cubo) en el árbol y contiene toda la lógica recursiva.

-   **Atributos Clave**:
    -   `centro` y `mitad`: Definen el volumen cúbico del nodo en el espacio 3D.
    -   `puntos`: Una lista que almacena los puntos 3D si el nodo es una **hoja** (no tiene hijos).
    -   `hijos`: Una lista de 8 `NodoOctree` si es un **nodo interno**.
    -   `max_puntos` y `max_profundidad`: Controlan cuándo un nodo hoja debe subdividirse.

-   **Método de Inserción (`insertar`)**:
    1.  Si el nodo ya tiene hijos, determina a cuál de los 8 octantes pertenece el punto (usando `_hijo_adecuado`) y le delega la inserción de forma recursiva.
    2.  Si es un nodo hoja, simplemente añade el punto a su lista `puntos`.
    3.  **La magia de la subdivisión**: Después de añadir un punto, si el número de puntos en la hoja supera `max_puntos` y no se ha alcanzado `max_profundidad`, el nodo se subdivide llamando a `_subdividir()`. Luego, re-inserta todos los puntos que contenía en sus nuevos hijos, convirtiéndose en un nodo interno.

-   **Método de Búsqueda (`buscar_rango`)**:
    Este método encuentra eficientemente todos los puntos dentro de un radio esférico.
    1.  **Poda del Árbol**: Primero comprueba si el cubo del nodo se intersecta con la esfera de búsqueda (`_intersecta_esfera`). Si no hay intersección, **descarta esta rama completa del árbol** y todos sus descendientes, lo que acelera masivamente la búsqueda.
    2.  **Recursión**: Si hay intersección y es un nodo interno, llama recursivamente a `buscar_rango` en sus 8 hijos.
    3.  **Comprobación Final**: Si es un nodo hoja, itera sobre su pequeña lista de puntos y realiza una comprobación de distancia exacta (`np.linalg.norm`) para ver cuáles están realmente dentro del radio.

#### 2. La Clase `Octree`

Esta clase actúa como una envoltura o fachada (`Facade`) para simplificar el uso de la estructura de datos.

-   **Inicialización**: Al crear un objeto `Octree`, se inicializa automáticamente el nodo `raiz` que abarca todo el espacio de trabajo.
-   **Métodos Simplificados**: Proporciona métodos `insertar` y `buscar_rango` limpios que simplemente llaman a los métodos correspondientes del nodo raíz, ocultando la complejidad recursiva al usuario final.

### Utilidad de Conversión ([`src/convertor.py`](src/convertor.py))

Para facilitar la interoperabilidad con otras herramientas de software 3D, el proyecto incluye un script de utilidad para la conversión de formatos de malla. Este es una herramienta de línea de comandos que utiliza la poderosa biblioteca `trimesh` para convertir archivos `.glb` a otros formatos comunes.

#### Características Principales

-   **Carga GLB (`load_glb`)**: El script está diseñado para manejar archivos `.glb` complejos. Si el archivo contiene una escena con múltiples mallas separadas, la función las **fusiona automáticamente** en un único objeto `trimesh.Trimesh`, simplificando el proceso de conversión.
-   **Soporte Multiformato (`export_mesh`)**: Permite la exportación a los formatos 3D más comunes:
    -   `OBJ` (con materiales, si están presentes)
    -   `PLY` (ideal para nubes de puntos y mallas simples)
    -   `STL` (común en impresión 3D)
-   **Interfaz de Línea de Comandos (`argparse`)**: Proporciona una interfaz de usuario flexible y fácil de usar para especificar el archivo de entrada, los formatos de salida deseados y un directorio de destino.

### Interfaz web: Cambios del frontend de Hunyuan3D-2 ([`src/app.py`](src/app.py))

#### `build_model_viewer_html`

Esta función genera dinámicamente el código HTML necesario para mostrar un modelo 3D (`.glb`) en la interfaz.

Su funcionamiento es el siguiente:
1.  **Lee una Plantilla:** Carga un archivo `template.html` que contiene el código para el visor de modelos `<model-viewer>`.
2.  **Inyecta Datos:** Reemplaza marcadores de posición en la plantilla (como `#src#`, `#height#`) con la ruta del archivo `.glb` específico que se acaba de generar y las dimensiones deseadas.
3.  **Crea un Archivo HTML:** Guarda este HTML personalizado en una carpeta temporal.
4.  **Retorna un `<iframe>`:** La función devuelve una etiqueta `<iframe>` de HTML. Este `<iframe>` carga el archivo HTML recién creado desde una ruta estática (`/static/...`), permitiendo que el visor 3D se **incruste directamente** dentro de la interfaz de Gradio como si fuera un componente nativo.

#### Gradio

**Gradio** es una biblioteca de Python que permite crear rápidamente interfaces de usuario web interactivas para modelos de machine learning, sin necesidad de escribir código de front-end (HTML, CSS, JavaScript).

En este script, se utiliza de la siguiente manera:
-   **`gr.Blocks`**: Actúa como un lienzo donde se organizan todos los componentes visuales (botones, sliders, áreas para subir imágenes, etc.).
-   **Componentes**: Se usan elementos como `gr.Button`, `gr.Image`, `gr.Slider` y `gr.HTML` para construir la interfaz.
-   **Eventos (`.click()`)**: El método `.click()` es el núcleo de la interactividad. Conecta un componente de la interfaz (ej. un botón) con una función de Python (ej. `shape_generation`). Cuando el usuario interactúa con el componente en el navegador, Gradio ejecuta la función asociada en el backend, pasándole los valores de los campos de entrada y mostrando los resultados en los campos de salida.

### `build_app()`

Esta función es la **constructora de la interfaz de usuario**. Su responsabilidad es definir, organizar y conectar todos los elementos visuales de la aplicación Gradio.

1.  **Define la Estructura**: Utiliza `gr.Blocks`, `gr.Row`, `gr.Column` y `gr.Tab` para crear el diseño visual de la página web.
2.  **Crea los Controles**: Instancia todos los componentes interactivos que el usuario verá, como los cuadros para cargar imágenes, el campo de texto para el *prompt*, los botones "Generar", y los sliders para las opciones avanzadas (seed, pasos de inferencia, etc.).
3.  **Conecta la Lógica**: Establece los "oyentes de eventos" con los métodos `.click()` y `.change()`. Por ejemplo, asocia el botón "Generar" con la función `shape_generation`, especificando qué componentes de la UI actúan como entradas y cuáles como salidas.
4.  **Devuelve la App**: Al final, la función retorna el objeto `demo` de Gradio, que representa la aplicación web completamente construida y lista para ser montada en un servidor.

###  El Levantamiento con Uvicorn

**Uvicorn** es un servidor web ASGI (Asynchronous Server Gateway Interface) de alto rendimiento. Es necesario para "servir" la aplicación web creada con FastAPI y Gradio, haciéndola accesible a través de un navegador.

La línea `uvicorn.run(app, host=args.host, port=args.port, workers=1)` realiza lo siguiente:
-   **Inicia el Servidor**: Pone en marcha un proceso de servidor web.
-   **Carga la Aplicación**: Le indica a Uvicorn que cargue el objeto `app` (que es una aplicación FastAPI que tiene montada la interfaz de Gradio).
-   **Expone la Aplicación**: Hace que la interfaz web sea accesible en la red en el `host` y `port` especificados en los argumentos (por ejemplo, `http://0.0.0.0:8080`). Una vez que esta línea se ejecuta, puedes abrir tu navegador y navegar a esa dirección para usar la aplicación.

## Resultados

CHAMO genera avatares visualmente ricos y detallados, compatibles con formatos y plataformas estándar. La calidad visual iguala a la de técnicas más complejas y costosas, validando la efectividad del pipeline híbrido propuesto.

-   **Avatar 3D de alta calidad** generado desde una sola imagen.
-   **Tiempo de inferencia inferior a 1 minuto** en una GPU NVIDIA RTX 4090.
-   **Exportación en formatos** `PLY`, `OBJ`, `GLB`, con texturas en `PNG`.
-   **Compatible** con Blender, Unity y Unreal Engine.

#### Galería de Ejemplos



![Video rotatorio](images/aplicaciones_1_2.png)


![Video rotatorio](images/aplicaciones_1_3.png)



![Video rotatorio](images/aplicaciones_2_1.png)



![Video rotatorio](images/aplicaciones_2_3.png)



![Video rotatorio](images/aplicaciones_3_2.png)



![Video rotatorio](images/aplicaciones_4_1.png)

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

