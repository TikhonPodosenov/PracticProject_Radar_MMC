from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from ultralytics import YOLO
import cv2
import os

class Ui_PersonDetecion(object):
    path = ""

    def setupUi(self, PersonDetecion):
        PersonDetecion.setObjectName("PersonDetecion")
        PersonDetecion.resize(481, 275)
        PersonDetecion.setStyleSheet("{background-image: url(./background/camera.jpg);}")
        self.centralwidget = QtWidgets.QWidget(PersonDetecion)
        self.centralwidget.setObjectName("centralwidget")
        self.buttonSelect = QtWidgets.QPushButton(self.centralwidget)
        self.buttonSelect.setGeometry(QtCore.QRect(60, 30, 371, 41))
        self.buttonSelect.setObjectName("buttonSelect")
        self.buttonDetect = QtWidgets.QPushButton(self.centralwidget)
        self.buttonDetect.setGeometry(QtCore.QRect(250, 160, 181, 41))
        self.buttonDetect.setEnabled(False)
        self.buttonDetect.setObjectName("buttonDetect")
        self.textPuth = QtWidgets.QTextEdit(self.centralwidget)
        self.textPuth.setText("File - viewing the original media file\nDetect - viewing a processed media file")
        self.textPuth.setGeometry(QtCore.QRect(60, 80, 371, 71))
        self.textPuth.setObjectName("textPuth")
        self.buttonFile = QtWidgets.QPushButton(self.centralwidget)
        self.buttonFile.setGeometry(QtCore.QRect(60, 160, 181, 41))
        self.buttonFile.setEnabled(False)
        self.buttonFile.setObjectName("buttonFile")
        PersonDetecion.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(PersonDetecion)
        self.statusbar.setObjectName("statusbar")
        PersonDetecion.setStatusBar(self.statusbar)

        self.retranslateUi(PersonDetecion)
        QtCore.QMetaObject.connectSlotsByName(PersonDetecion)
        self.add_function()

    def add_function(self):
        self.buttonSelect.clicked.connect(lambda: self.openFileDialog())
        self.buttonDetect.clicked.connect(lambda: self.getResult(self.path))
        self.buttonFile.clicked.connect(lambda: self.showMediaFile(self.path))

    def retranslateUi(self, PersonDetecion):
        _translate = QtCore.QCoreApplication.translate
        PersonDetecion.setWindowTitle(_translate("PersonDetecion", "PersonDetecion"))
        self.buttonSelect.setText(_translate("PersonDetecion", "Select file"))
        self.buttonDetect.setText(_translate("PersonDetecion", "Detect"))
        self.buttonFile.setText(_translate("PersonDetecion", "File"))

    def showMediaFile(self, path):
        base, ext = os.path.splitext(path)
        if ext.lower() == '.mp4':
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                success, img = cap.read()
                img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
                cv2.imshow("Press 'esc' to exit", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            cap.release()
        else:
            img = cv2.imread(path)
            img = cv2.resize(img, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)))
            cv2.imshow("Person detection", img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detectImage(self, model, path):
        try:
            results = model.predict(source=path)
            #test data
            num_objects = 0
            for r in results:
                boxes = r.boxes.data
                num_objects = len(boxes)

            img = results[0].plot()
            img = cv2.resize(img, (int(img.shape[1]*0.2), int(img.shape[0]*0.2)))
            cv2.imshow("Person detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {e}")
        return int(num_objects)

    def detectVideo(self, model, path):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.predict(frame, augment=True, imgsz=960, stream_buffer=True)
                annotated_frame = results[0].plot()
                cv2.resize(annotated_frame,(0, 0), fx=0.75, fy=0.75)
                cv2.imshow("Press 'esc' to exit", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def openFileDialog(self):
        filename, filetype = QFileDialog.getOpenFileName(
                    PersonDetecion,
                    'Open file',
                    '.',
                    "Image files(*.jpg *.png);; Video files(*.mp4)")
        if filename:
            self.path = filename
            self.textPuth.setText(filename)
            self.buttonFile.setEnabled(True)
            self.buttonDetect.setEnabled(True)

    def getResult(self, path):
        model = YOLO("weight_main_30.pt")
        base, ext = os.path.splitext(path)
        if ext.lower() == '.mp4':
            self.detectVideo(model, path)
        else:
            self.detectImage(model, path)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyleSheet(stylesheet)
    PersonDetecion = QtWidgets.QMainWindow()
    ui = Ui_PersonDetecion()
    ui.setupUi(PersonDetecion)
    PersonDetecion.show()
    sys.exit(app.exec_())
