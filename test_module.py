import unittest
from form import Ui_PersonDetecion as person
from ultralytics import YOLO


class TestForm(unittest.TestCase):
    model = YOLO("weight_main_30.pt")
    path_image = "./test/02_1_000938.JPG"
    path_test = "C:\PycharmProjects\practicProject\test\02_1_000938.JPG"
    def test_detectImage(self):
        self.assertEqual(person.detectImage(person, self.model, self.path_image), 3)
