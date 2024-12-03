# Towards efficient Deepfake detection with Self Supervised Learning

The concept of *Deepfakes*, which emerged just recently, commonly refers to fabricated digital content, created by either altering existing footage or generating entirely new visuals from scratch. 
The term `deep' signifies the use of deep learning algorithms, a powerful branch of Machine Learning (ML) with applications in computer vision, speech recognition, and various other fields. Deep learning has achieved impressive results, sometimes even exceeding human experts.
While deepfakes were once easier to detect, advancements in artificial intelligence (AI) and increased computing power have fueled the development of highly realistic image-generation techniques.
This has led to the creation of user-friendly apps that can generate deepfakes in real time. The spread of these fabricated videos and images on social media raises significant concerns about misinformation and the trustworthiness of online information. 

### Synthetic faces & Human perception
Spotting deepfakes is even more challenging in the face generation realm. Even trained human observers struggle, with studies like \cite{TestingHumanAbilityDetectDeepfakeImages} showing human accuracy varies between 30\% and 85\% depending on the image, averaging around 62\%. Moreover, the article \cite{TestingHumanAbilityDetectDeepfakeImages} shows that for one image out of five the accuracy does not reach 50\%, making a coin toss preferable to the human judgment. Results are similar in \cite{DBLP:journals/corr/abs-2106-07226}, where human performances have been tested against technologies from more and less recent years. Here, humans labeled synthetic images as real 68\% of the time, while only correctly identifying real images 52\% of the time. These findings highlight the urgent need for automated detection tools, especially in areas relying on authenticity, like user verification and combating the spread of misinformation. 



### The effects of deepfakes' diffusion on social media
To address this challenge, researchers created AI models that detect deepfakes by analyzing patterns and artifacts left behind during the generation process \cite{gragnaniello2021gan}. However, social media sharing often degrades these traces by compressing with private algorithms, specific to each platform \cite{10007988}. This process makes detection difficult, hindering model performance depending on the platform the image has been compressed by.

To combat this issue in real-world social media settings, a dataset called `TrueFace' \cite{10007988} was created. It contains real and fake images from Facebook, Telegram, and Twitter, allowing researchers to train models on images subject to social media compression. Ongoing research focuses on developing techniques that enable models to generalize across different platforms, regardless of their sharing history.

## Machine Learning on Tiny Devices

The Internet of Things (IoT) is a vast network of interconnected smart devices. To effectively process the massive amount of data they produce, we need to shift intelligence from centralized cloud servers to the edge of the network, closer to devices and sensors. However, edge processing presents unique challenges due to resource constraints such as limited memory, communication bandwidth, and battery power, as shown in \cite{10377703}. These limitations make it difficult to run traditional AI algorithms on peripheral devices like sensor nodes.

To overcome these challenges and bring AI capabilities to the edge, Tiny Machine Learning (TinyML) has emerged \cite{Kallimani_2023}. TinyML focuses on developing lightweight machine learning models and algorithms that can operate efficiently within the constraints of edge devices.
This technology offers numerous advantages in processing sensor data on low-power embedded devices. Since no connectivity is required for inference, data can be processed in real-time directly on the device, avoiding the need to transfer raw data to third parties for processing. This eliminates delays due to waiting for unnecessary transmission times and significantly reduces latency.

To understand the impact of low response latency, consider the continuous monitoring of machines in industrial settings to predict problems and potential failures. This type of application can provide a timely response to damage, reducing maintenance costs, risks of failure, and downtime, while improving performance and efficiency. 

Embedded devices that support TinyML algorithms require a very small amount of power (in the order of milliwatts), which allows them to operate for long periods without needing to be charged if equipped with a battery (see example in \cite{paissan2021phinets}), or with minimal energy consumption if powered. Research for reducing power requirements does not only aim at bringing AI to microcontrollers. The training of big Machine Learning Models from the most powerful companies in the world like Microsoft, Meta, or Google, requires a lot of power for their training and deployment. Currently, investors are readily backing research in this direction, and only major corporations can manage the financial burden. But as models demand ever-greater resources, the long-term sustainability of this approach remains unclear. Recent studies tried to benchmark the CO$_2$ emissions of big language models like BLOOM or GPT-3 \cite{carbonFootprintofMLTraining, luccioni2022estimatingBLOOM}. Additionally, the competition among big techs for the best model is aggravating the situation. For example, ChatGPT from OpenAI and Gemini from Google are both computationally expensive and potentially contribute to similar environmental impacts, despite fulfilling similar purposes. This redundancy could lead to increased energy consumption without significant advancements in functionality.

## Deepfake Detection

Moving deepfake detection to edge devices offers several benefits. First, it expands the capabilities of microcontrollers (MCUs) with a new, challenging task. Secondly, reducing the number of operations speeds up inference time (the time for the model to analyze and judge an image). Faster inference brings us closer to real-time applications. While frame-by-frame deepfake analysis for videos may not seem immediately necessary, it's certainly not a drawback.

A significant advantage is that lower deployment requirements allow devices to perform inference locally \cite{paissan2021phinets}. For instance, a small network on a smartphone could verify content authenticity without using bandwidth. More broadly, faster inference is crucial given the vast number of images uploaded online every second. Unofficial statistics suggest Instagram sees about a thousand image uploads per second, 34\% of which involve human subjects. If the goal is to reality-check every image before posting, efficiency is paramount.


## Methods and results

This thesis addresses the critical need to adapt deepfake detection to real-world challenges. It focuses on two key strategies: enhancing model resilience against social media compression artifacts and enabling efficient deepfake detection on edge devices.

The first strategy leverages Self-Supervised Learning (SSL) to improve model robustness against distortions introduced by social media compression algorithms. By training models on unlabeled data to learn inherent structures and relationships, SSL enhances their ability to recognize deepfakes despite compression. Detailed exploration of SSL is provided in Section \ref{sec:learning_paradigms}.

Experimental results demonstrate that SSL significantly mitigates performance degradation on compressed images, narrowing the accuracy gap between pristine and manipulated content. However, this increased robustness may come at the cost of overall detection accuracy, highlighting the need for careful consideration of trade-offs based on specific application requirements.

The second strategy investigates lightweight neural network architectures, MicroNet \cite{li2021micronet} and PhiNet \cite{paissan2021phinets}, for on-device deepfake detection. These architectures prioritize a small parameter count and reduced computational complexity, essential for deployment on resource-constrained edge devices.

Research findings show that under controlled conditions, these compact models can achieve accuracy comparable to larger architectures like ResNet50 \cite{he2015deep}. For instance, MicroNet's accuracy on pre-social media images is only marginally lower than ResNet50's, while utilizing a fraction of the parameters. This finding holds significant promise for enabling real-time deepfake analysis directly on user devices, thereby potentially mitigating the spread of misleading content before it reaches wider audiences.