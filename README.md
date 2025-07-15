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



