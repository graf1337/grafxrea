import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Conv2D, BatchNormalization, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Input, Add, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
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

        self.conv1 = Conv2D(
            filters // 4, 1, strides=stride, padding='valid',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            filters // 4, 3, strides=1, padding='same',
            use_bias=False, kernel_initializer='he_normal'
        )
        self.bn2 = BatchNormalization()

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

# 4. ResNet50 архитектура
class ResNet50:
    def __init__(self, num_classes=10):  # 10 классов для синтетики
        self.num_classes = num_classes

    def build_model(self, input_shape=(32, 32, 3)):  # Меньший размер для быстрого обучения
        inputs = Input(shape=input_shape)

        x = Conv2D(
            64, 7, strides=2, padding='same', use_bias=False,
            kernel_initializer='he_normal', name='conv1'
        )(inputs)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu', name='conv1_relu')(x)
        x = MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

        for _ in range(3): x = BottleneckBlock(64)(x)
        x = BottleneckBlock(128, stride=2)(x); [BottleneckBlock(128)(x) for _ in range(3)]
        x = BottleneckBlock(256, stride=2)(x); [BottleneckBlock(256)(x) for _ in range(5)]
        x = BottleneckBlock(512, stride=2)(x); [BottleneckBlock(512)(x) for _ in range(2)]

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = Dense(
            self.num_classes, activation='softmax',
            kernel_initializer='he_normal', name='fc1000'
        )(x)

        return Model(inputs, outputs, name='resnet50')

# 🔥 ГРАФИК МЕТРИК ОБУЧЕНИЯ
def plot_training_metrics(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], 'b-', linewidth=2, label='Train Loss')
    ax2.plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ График метрик сохранён: training_metrics.png")

# Визуализация активаций
def visualize_network_work(model):
    target_layer_names = ['conv1', 'conv1_relu', 'pool1', 'avg_pool']
    target_layers = []
    for name in target_layer_names:
        try:
            target_layers.append(model.get_layer(name))
        except:
            continue

    for layer in model.layers:
        if isinstance(layer, BottleneckBlock):
            target_layers.append(layer)
            break

    activation_model = Model(inputs=model.input, outputs=[layer.output for layer in target_layers])
    img = tf.random.normal((1, 32, 32, 3))
    activations = activation_model(img, training=False)

    n = len(target_layers)
    plt.figure(figsize=(4 * n, 4))

    for i, (layer, act) in enumerate(zip(target_layers, activations), 1):
        plt.subplot(1, n, i)
        act_np = act.numpy()
        if act_np.ndim == 4:
            plt.imshow(act_np[0, :, :, 0], cmap='viridis')
        elif act_np.ndim == 2:
            plt.plot(act_np[0])
        plt.title(layer.name)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resnet50_work_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# 🔥 ОСНОВНОЙ ЗАПУСК С ГРАФИКОМ МЕТРИК
if __name__ == "__main__":
    print("🚀 Создание ResNet-50...")
    
    # Модель (меньший input_shape для быстрого обучения)
    resnet = ResNet50(num_classes=10)
    model = resnet.build_model(input_shape=(32, 32, 3))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📊 Параметров: {model.count_params():,}")
    model.summary()
    
    # 🔥 СИНТЕТИЧЕСКИЕ ДАННЫЕ И ОБУЧЕНИЕ
    print("\n🔥 Создание синтетических данных...")
    x_train = tf.random.normal((2000, 32, 32, 3))
    y_train = tf.random.uniform((2000,), 0, 10, dtype=tf.int32)
    x_val = tf.random.normal((500, 32, 32, 3))
    y_val = tf.random.uniform((500,), 0, 10, dtype=tf.int32)
    
    print("🔥 Обучение модели (10 эпох)...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # 🔥 3 ВИЗУАЛИЗАЦИИ
    print("\n🎨 Создание визуализаций...")
    
    # 1. Архитектура
    plot_model(model, to_file='resnet50_architecture.png', show_shapes=True, dpi=96)
    print("✅ 1. resnet50_architecture.png")
    
    # 2. Активации
    visualize_network_work(model)
    print("✅ 2. resnet50_work_visualization.png")
    
    # 3. ГРАФИК МЕТРИК
    plot_training_metrics(history)
    print("✅ 3. training_metrics.png")
    
    print("\n🎉 ВСЕ 3 ГРАФИКА СОЗДАНЫ!")
