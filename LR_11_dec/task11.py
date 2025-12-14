import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Input, Add, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np


# 1. ResidualBlock с правильной skip-логикой
class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride

        self.conv1 = Conv2D(
            filters, kernel_size, strides=stride, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters, kernel_size, strides=1, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn2 = BatchNormalization()

        self.shortcut = None

    def build(self, input_shape):
        super().build(input_shape)
        in_channels = input_shape[-1]
        if self.stride != 1 or in_channels != self.filters:
            self.shortcut = [
                Conv2D(
                    self.filters, 1, strides=self.stride, padding='valid',
                    use_bias=False, kernel_initializer='he_normal'
                ),
                BatchNormalization()
            ]

    def call(self, x, training=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.shortcut is not None:
            for layer in self.shortcut:
                if isinstance(layer, BatchNormalization):
                    identity = layer(identity, training=training)
                else:
                    identity = layer(identity)

        out = Add()([out, identity])
        out = Activation('relu')(out)
        return out


# 2. BottleneckBlock (1x1 -> 3x3 -> 1x1) с адаптацией skip connection
class BottleneckBlock(Layer):
    def __init__(self, filters, stride=1):
        super(BottleneckBlock, self).__init__()
        self.filters = filters
        self.stride = stride

        # 1x1 (уменьшение)
        self.conv1 = Conv2D(
            filters // 4, 1, strides=stride, padding='valid',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn1 = BatchNormalization()

        # 3x3
        self.conv2 = Conv2D(
            filters // 4, 3, strides=1, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn2 = BatchNormalization()

        # 1x1 (увеличение)
        self.conv3 = Conv2D(
            filters, 1, strides=1, padding='valid',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn3 = BatchNormalization()

        self.shortcut = None

    def build(self, input_shape):
        super().build(input_shape)
        in_channels = input_shape[-1]
        if self.stride != 1 or in_channels != self.filters:
            self.shortcut = [
                Conv2D(
                    self.filters, 1, strides=self.stride, padding='valid',
                    use_bias=False, kernel_initializer='he_normal'
                ),
                BatchNormalization()
            ]

    def call(self, x, training=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.shortcut is not None:
            for layer in self.shortcut:
                if isinstance(layer, BatchNormalization):
                    identity = layer(identity, training=training)
                else:
                    identity = layer(identity)

        out = Add()([out, identity])
        out = Activation('relu')(out)
        return out


# 4. ResNet50 архитектура: Conv7x7 -> MaxPool -> (3,4,6,3) bottleneck-стейджи -> GAP -> Dense
class ResNet50:
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes

    def build_model(self, input_shape=(224, 224, 3)):
        inputs = Input(shape=input_shape)

        # Стартовый блок
        x = Conv2D(
            64, 7, strides=2, padding='same', use_bias=False,
            kernel_initializer='he_normal', name='conv1'
        )(inputs)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu', name='conv1_relu')(x)
        x = MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

        # Stage 1: 3 bottleneck блока с 64 фильтрами
        for _ in range(3):
            x = BottleneckBlock(64)(x)

        # Stage 2: 4 bottleneck блока с 128 фильтрами (stride=2 в первом)
        x = BottleneckBlock(128, stride=2)(x)
        for _ in range(3):
            x = BottleneckBlock(128)(x)

        # Stage 3: 6 bottleneck блоков с 256 фильтрами (stride=2 в первом)
        x = BottleneckBlock(256, stride=2)(x)
        for _ in range(5):
            x = BottleneckBlock(256)(x)

        # Stage 4: 3 bottleneck блока с 512 фильтрами (stride=2 в первом)
        x = BottleneckBlock(512, stride=2)(x)
        for _ in range(2):
            x = BottleneckBlock(512)(x)

        # Голова
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = Dense(
            self.num_classes, activation='softmax',
            kernel_initializer='he_normal', name='fc1000'
        )(x)

        model = Model(inputs, outputs, name='resnet50')
        return model


# Визуализация работы сети: активации нескольких слоёв
def visualize_network_work(model):
    from tensorflow.keras.models import Model

    # Выбираем несколько характерных слоёв по имени
    target_layer_names = ['conv1', 'conv1_relu', 'pool1', 'avg_pool']
    target_layers = []
    for name in target_layer_names:
        try:
            target_layers.append(model.get_layer(name))
        except ValueError:
            print(f"Слой {name} не найден, пропускаем")

    # Добавим один bottleneck-блок по типу
    for layer in model.layers:
        if isinstance(layer, BottleneckBlock):
            target_layers.append(layer)
            break

    # Модель для вывода активаций
    activation_model = Model(
        inputs=model.input,
        outputs=[layer.output for layer in target_layers]
    )

    # Тестовый вход
    img = tf.random.normal((1, 224, 224, 3))
    activations = activation_model(img, training=False)

    n = len(target_layers)
    plt.figure(figsize=(4 * n, 4))

    for i, (layer, act) in enumerate(zip(target_layers, activations), start=1):
        plt.subplot(1, n, i)
        act_np = act.numpy()

        # Безопасная обработка размерностей
        if act_np.ndim == 4:
            fmap = act_np[0, :, :, 0]
            plt.imshow(fmap, cmap='viridis')
        elif act_np.ndim == 2:
            vec = act_np[0]
            plt.plot(vec)
        elif act_np.ndim == 3:
            fmap = act_np[0, :, 0]
            plt.imshow(np.expand_dims(fmap, -1), cmap='viridis')
        else:
            plt.text(0.1, 0.5, f"shape: {act_np.shape}")
            plt.axis('off')
            continue

        plt.title(layer.name)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resnet50_work_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Визуализация активаций сохранена в resnet50_work_visualization.png")


# Основной запуск
if __name__ == "__main__":
    resnet = ResNet50(num_classes=1000)
    model = resnet.build_model(input_shape=(224, 224, 3))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Визуализация архитектуры
    plot_model(
        model, to_file='resnet50_architecture.png',
        show_shapes=True, show_layer_names=True, dpi=96
    )
    print("Архитектура сохранена в resnet50_architecture.png")

    # Визуализация работы сети (активации)
    visualize_network_work(model)

    # Тестовый forward pass
    dummy_input = tf.random.normal((1, 224, 224, 3))
    preds = model(dummy_input)
    print("Форма выхода:", preds.shape)
