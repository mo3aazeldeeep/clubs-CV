import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def extract_features(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return des



class BOVW:
    def __init__(self, train_path, test_path, clusters=120):
        self.train_path = train_path
        self.test_path = test_path
        self.clusters = clusters

        self.kmeans = MiniBatchKMeans(n_clusters=clusters)
        self.svm = SVC(kernel='rbf', C=10, gamma='scale')  
        self.scaler = StandardScaler()

        self.label_map = {}
        self.idf = None


    def load_dataset(self, path):
        data = {}
        class_names = sorted(os.listdir(path))

        for idx, cls in enumerate(class_names):
            class_path = os.path.join(path, cls)

            if os.path.isdir(class_path):
                images = []

                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path)

                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)

                data[cls] = images
                self.label_map[idx] = cls

        return data


    def get_descriptors(self, data):
        descriptor_list = []
        labels = []

        for label, (cls, images) in enumerate(data.items()):
            for img in images:
                des = extract_features(img)

                if des is not None:
                    descriptor_list.append(des)
                    labels.append(label)

        return descriptor_list, labels


    def stack_descriptors(self, descriptor_list):
        vstack = descriptor_list[0]

        for desc in descriptor_list[1:]:
            vstack = np.vstack((vstack, desc))

        return vstack


    def apply_tfidf(self, X, is_train=False):
        if is_train:
            df = np.sum(X > 0, axis=0)
            N = X.shape[0]
            self.idf = np.log((N + 1) / (df + 1))

        return X * self.idf


    def train(self):
        print("Loading train data...")
        train_data = self.load_dataset(self.train_path)

        descriptor_list, train_labels = self.get_descriptors(train_data)

        print("Stacking descriptors...")
        all_desc = self.stack_descriptors(descriptor_list)

        print("Clustering...")
        self.kmeans.fit(all_desc)

        print("Building vocabulary...")
        X_train = np.zeros((len(descriptor_list), self.clusters))

        for i in range(len(descriptor_list)):
            desc = descriptor_list[i]
            clusters = self.kmeans.predict(desc)

            for c in clusters:
                X_train[i][c] += 1

        X_train = self.apply_tfidf(X_train, True)
        X_train = self.scaler.fit_transform(X_train)

        print("Training SVM...")
        self.svm.fit(X_train, train_labels)


    def test(self):
        print("Loading test data...")
        test_data = self.load_dataset(self.test_path)

        descriptor_list, test_labels = self.get_descriptors(test_data)

        X_test = np.zeros((len(descriptor_list), self.clusters))

        for i in range(len(descriptor_list)):
            desc = descriptor_list[i]

            if desc is not None:
                clusters = self.kmeans.predict(desc)
                for c in clusters:
                    X_test[i][c] += 1

        X_test = self.apply_tfidf(X_test)
        X_test = self.scaler.transform(X_test)

        preds = self.svm.predict(X_test)

        print("\nAccuracy:", accuracy_score(test_labels, preds))
        print("\nClassification Report:\n", classification_report(test_labels, preds))
        print("\nConfusion Matrix:\n", confusion_matrix(test_labels, preds))

        return test_data, preds, test_labels


    def show_batch(self, test_data, preds, labels, batch_size=5):
        images = []
        i = 0

        for cls, imgs in test_data.items():
            for img in imgs:
                if i >= len(preds):
                    break
                images.append(img)
                i += 1

        for start in range(0, len(images), batch_size):
            plt.figure(figsize=(15, 4))

            for i in range(batch_size):
                idx = start + i
                if idx >= len(images):
                    break

                img = images[idx]
                pred = preds[idx]
                true = labels[idx]

                correct = (pred == true)
                color = 'green' if correct else 'red'

                plt.subplot(1, batch_size, i+1)
                plt.imshow(img)
                plt.title(f"{'OK' if correct else 'WRONG'}\nP:{self.label_map[pred]}\nT:{self.label_map[true]}")
                plt.axis('off')

                for spine in plt.gca().spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)

            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(BASE_DIR, "train")
    test_path = os.path.join(BASE_DIR, "test")

    model = BOVW(train_path, test_path, clusters=200)
    model.train()

    test_data, preds, labels = model.test()
    model.show_batch(test_data, preds, labels)