import cv2
from kivy.event import EventDispatcher
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivymd.app import MDApp
from kivy.properties import ObjectProperty, StringProperty
from kivy.core.window import Window
from numba.core.types.misc import Object
from spectrogram import create_spectrograms, create_spectrograms_for_classif
from tensorflow.keras.preprocessing.image import img_to_array
from kivy.uix.scatter import Scatter
import numpy
from tensorflow.keras import models
from kivy.uix.screenmanager import Screen
from kivymd.uix.list import OneLineListItem
import threading
from kivy.clock import Clock, mainthread
import os
import glob
import io
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image
from plyer import filechooser
import skimage
import matplotlib.pyplot as plt
from kivy.uix.settings import SettingsWithSidebar
from settings import settings_json
from kivy.config import Config, ConfigParser


model = None
model_name = ""

committee = []
model_names = []

genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
          'Instrumental', 'International', 'Pop', 'Rock']

available_models = ['cnn', 'crnn', 'crnn2', 'crnn3', 'crnn4',
                    'crnn5', 'crnn6', 'crnn7', 'crnn8', 'prcnn']


def majority(votes):
    result = numpy.array([0.]*8)
    for vote in votes:
        vote = numpy.zeros_like(result)
        vote[numpy.argmax(vote)] = 1
        result += vote
    return result


def naive(votes):
    result = numpy.array([0.]*8)
    for vote in votes:
        result += vote
    return result


def kApproval(votes):
    result = numpy.array([0.]*8)
    for sample in votes:
        vote = numpy.zeros_like(result)
        vote[sample.argsort()[-1]] = 3
        vote[sample.argsort()[-2]] = 2
        vote[sample.argsort()[-3]] = 1
        result += vote
    return result


vote = {
    'majority': majority,
    'naive': naive,
    'kApproval': kApproval
}


def get_kivy_image_from_bytes(image_bytes, file_extension):
    buf = io.BytesIO(image_bytes)
    cim = CoreImage(buf, ext=file_extension)
    return Image(texture=cim.texture, nocache=True)


class obj():
    pass


def classify(file, spinner, output, classifier):
    global model_name
    global model
    spinner.active = True
    if (model_name != classifier):
        model = models.load_model('./models/fma.' + classifier)
        model_name = classifier
    input = []
    images = create_spectrograms_for_classif(file)
    for image in images:
        input_arr = img_to_array(image)
        input_arr /= 255.
        input_arr = numpy.array(input_arr)
        input.append(input_arr)
    predictions = numpy.array(model.predict(numpy.array(input)))
    result = numpy.array([0]*8)
    for prediction in predictions:
        prediction = prediction / numpy.sum(prediction)
        result = numpy.add(result, prediction)
    result = result / numpy.sum(result)
    output.vector = result
    spinner.active = False


def classify_committee(file, spinner, output, model_list, voting_method):
    global committee
    global model_names
    global vote
    spinner.active = True
    if model_names != model_list:
        for model in model_list:
            print(model)
            committee.append(
                models.load_model('./models/fma.' + model)
            )
    input = []
    images = create_spectrograms_for_classif(file)
    for image in images:
        input_arr = img_to_array(image)
        input_arr /= 255.
        input_arr = numpy.array(input_arr)
        input.append(input_arr)
    votes = []
    for model in committee:
        predictions = numpy.array(model.predict(numpy.array(input)))
        result = numpy.array([0]*8)
        for prediction in predictions:
            result = numpy.add(result, prediction)
        votes.append(result)

    summed = vote[voting_method](votes)
    summed = summed / numpy.sum(summed)
    output.vector = summed
    spinner.active = False


class ClassifiersApp(MDApp, EventDispatcher):
    dropFunctions = []
    images = None

    # StringProperty(os.getcwd())
    folder = StringProperty("F:/classifiers/fma_small/000")

    def build(self):
        self.settings_cls = SettingsWithSidebar
        # self.theme_cls.primary_palette = "Gray"
        self.theme_cls.theme_style = "Dark"
        Window.bind(on_dropfile=self._on_file_drop)
        self.use_kivy_settings = False
        # setting = self.config.get('options')
        return

    def build_config(self, config):
        config.read('classifiers.ini')

    def build_settings(self, settings):
        settings.add_json_panel('Settings',
                                self.config,
                                data=settings_json)

    def on_config_change(self, config, section,
                         key, value):
        pass

    def bindFileDrop(self, f):
        self.dropFunctions.append(f)

    def _on_file_drop(self, window, file_path):
        if (file_path.decode('utf-8').lower().endswith(('.mp3', '.wav'))):
            self.file = file_path.decode('utf-8')
            self.images = None
        for f in self.dropFunctions:
            f()


class SpectrogramsWidget(BoxLayout):
    loading = False
    folder = "F:/classifiers/fma_small/000"  # os.getcwd()
    result = obj()
    file = ""
    value = 100

    def init_widget(self, *args):
        self.ids.list.clear_widgets()
        # os.chdir(self.folder)
        array = glob.glob(self.folder + "/*.mp3") + \
            glob.glob(self.folder + "/*.wav")
        for file in array:
            widget = OneLineListItem(text=os.path.basename(file))
            widget.bind(on_press=self.press)
            self.ids.list.add_widget(
                widget
            )
        self.list = self.ids.images

    def press(self, item):
        self.file = item.text
        self.generate()

    def __init__(self, **kwargs):
        super(SpectrogramsWidget, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()
        Clock.schedule_once(self.init_widget, 0)
        self.app.bind(folder=self.folder_changed)
        self.result.img = []

    def folder_changed(self, instance, value):
        self.folder = self.app.folder
        self.init_widget(None)

    def active_change(self, instance, value):
        if(value == False):
            self.ids.images.clear_widgets()
            self.ids.images.height = 0
            for img in self.result.img:
                success, encoded_image = cv2.imencode('.png', img)
                img = get_kivy_image_from_bytes(encoded_image, "png")
                img.allow_stretch = True
                img.size_hint_x = 1
                self.ids.images.add_widget(img)
                self.ids.images.height += 100

    def generate(self):
        self.ids.images.clear_widgets()
        self.spinner.active = True
        self.ids.spinner.bind(active=self.active_change)
        thread = threading.Thread(
            target=create_spectrograms, args=(self.app.folder + "/" + self.file, self.result, self.list, self.spinner))
        thread.start()

    def save(self):
        path = filechooser.choose_dir()
        counter = 0
        if len(path) > 0:
            for img in self.result.img:
                skimage.io.imsave(path[0] + "/" + self.file +
                                  str(counter) + '.png', img)
                counter += 1


class App(Screen):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()
    file = StringProperty()
    folder = StringProperty()

    def __init__(self, **kwargs):
        super(App, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()
        self.app.dropFunctions.append(self._on_file_drop)
        self.folder = self.app.folder
        self.app.bind(folder=self.folder_changed)

    def folder_changed(self, instance, value):
        self.folder = self.app.folder

    def _on_file_drop(self):
        self.file = self.app.file.split('\\')[-1]


class ClassificationWidget(BoxLayout):
    loading = False
    folder = "F:/classifiers/fma_small/000"  # os.getcwd()
    prediction = obj()

    def init_widget(self, *args):
        self.ids.list.clear_widgets()
        # os.chdir(self.folder)
        array = glob.glob(self.folder + "/*.mp3") + \
            glob.glob(self.folder + "/*.wav")
        for file in array:
            widget = OneLineListItem(text=os.path.basename(file))
            widget.bind(on_press=self.press)
            self.ids.list.add_widget(
                widget
            )

    @mainthread
    def active_change(self, instance, value):
        if (value == False):
            plt.rcdefaults()
            plt.style.use('dark_background')

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.2)

            y_pos = numpy.arange(len(genres))
            ax.barh(y_pos, self.prediction.vector)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(genres)
            ax.invert_yaxis()
            ax.set_title(self.file)
            ax.set_xlabel('Prediction')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            cim = CoreImage(buf, ext='png')
            img = Image(texture=cim.texture, nocache=True)

            if os.path.isfile("./tmp.png"):
                os.remove("./tmp.png")
            self.ids.plot.clear_widgets()
            self.ids.plot.add_widget(img)

    def press(self, item):
        self.file = item.text
        self.classify()

    def __init__(self, **kwargs):
        super(ClassificationWidget, self).__init__(**kwargs)
        self.app = MDApp.get_running_app()
        Clock.schedule_once(self.init_widget, 0)
        self.app.bind(folder=self.folder_changed)
        self.prediction.vector = []

    def folder_changed(self, instance, value):
        self.folder = self.app.folder
        self.init_widget(None)

    def classify(self):
        method = self.app.config.get('configuration', 'method')
        classifier = self.app.config.get('configuration', 'model')
        self.ids.plot.clear_widgets()
        self.ids.spinner.bind(active=self.active_change)
        if method == 'classifier':
            thread = threading.Thread(
                target=classify, args=(self.folder + "/" + self.file, self.spinner, self.prediction, classifier.lower()))
        else:
            arr = []
            for name in available_models:
                if self.app.config.get('configuration', name) == "1":
                    arr.append(name)
            print(arr)
            method = self.app.config.get('configuration', 'voting')
            thread = threading.Thread(
                target=classify_committee, args=(self.folder + "/" + self.file, self.spinner, self.prediction, arr, method))
        thread.start()


class FileWidget(BoxLayout):
    text_input = StringProperty()

    def __init__(self, **kwargs):
        self.app = MDApp.get_running_app()
        Clock.schedule_once(self.init_widget, 0)
        return super().__init__(**kwargs)

    def init_widget(self, *args):
        fc = self.ids['filechooser']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_chooser, file_list_entry, *args):
        file_list_entry.children[0].color = (1.0, 1.0, 1.0, 1.0)  # File Names
        file_list_entry.children[1].color = (1.0, 1.0, 1.0, 1.0)  # Dir Names`

    def navigate(self):
        self.app.folder = self.text_input


class SettingsWidget(BoxLayout):
    pass


ClassifiersApp().run()
