from tensorflow.keras.models import Model

def extract_features(autoencoder, X):

    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("bottleneck").output
    )

    features = encoder.predict(X)

    return features