"""Окно ЛР 9. Формулы остаются в модулях программного рендера."""
import time
from pathlib import Path

import numpy as np

from .qt import QtCore as C, QtGui as G, QtWidgets as W
from .geometry import SOLIDS
from .generators import PROFILE_DEFAULT, FUNCTIONS, function_surface, read_profile, revolution, sphere
from .obj_io import load_obj, save_obj
from .scene import SceneObject, overlap_scene
from .renderer import Viewport
from .transforms import translation, scaling, rotation_axis, rotate_around_line, reflection, around_point

ROOT = Path(__file__).resolve().parents[2]
STYLE = """
QWidget { color:#e6eef8; font-size:13px }
QMainWindow,QWidget#root { background:#0c1320 }
QLabel#heading { font-size:23px; font-weight:700 }
QLabel#muted { color:#9aafc9 }
QGroupBox { border:1px solid #2b3e56; border-radius:8px; margin-top:15px; padding:12px 8px 8px; font-weight:600 }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px }
QPushButton,QComboBox,QAbstractSpinBox,QLineEdit,QPlainTextEdit {
  background:#18273a; border:1px solid #334b66; border-radius:5px; padding:5px; min-height:23px }
QPushButton:hover { background:#294664 }
QPushButton[primary="true"] { background:#265991; border-color:#4b7cb0; font-weight:600 }
QComboBox QAbstractItemView,QMenu { background:#18273a; selection-background-color:#265991 }
QTabWidget::pane { border:1px solid #293d56; border-radius:7px }
QTabBar::tab { background:#18273a; padding:8px 9px; margin-right:2px }
QTabBar::tab:selected { background:#295586 }
QScrollArea { border:0; background:#101b2a }
QWidget#tabbody { background:#101b2a }
QStatusBar,QMenuBar { background:#101b2a }
QMenuBar::item:selected { background:#265991 }
"""


def spin(value=0.,low=-1e4,high=1e4):
    widget = W.QDoubleSpinBox()
    widget.setDecimals(3)
    widget.setRange(low,high)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    widget.setSingleStep(.25)
    return widget


def count(value,low,high):
    widget = W.QSpinBox()
    widget.setRange(low,high)
    widget.setValue(value)
    return widget


def values(widgets):
    return [w.value() for w in widgets]


def note(text):
    label = W.QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("muted")
    return label


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная №9 — закраска и текстуры")
        self.resize(1240,800)
        self.setMinimumSize(920,620)
        self.setStyleSheet(STYLE)
        self.viewport = Viewport()
        self.opts = self.viewport.render_options
        self.items, self.selected, self.histories = [], 0, {}
        root = W.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = W.QHBoxLayout(root)
        preview = W.QVBoxLayout()
        self.heading = W.QLabel("Сфера")
        self.heading.setObjectName("heading")
        preview.addWidget(self.heading)
        preview.addWidget(self.viewport,1)
        self.metrics = note("")
        preview.addWidget(self.metrics)
        preview.addWidget(note("ЛКМ — орбита камеры  ·  Колесо — масштаб вида"))
        self.viewport.frame_ready.connect(self.metrics.setText)
        layout.addLayout(preview,1)
        self.tabs = W.QTabWidget()
        self.tabs.setMinimumWidth(421)
        self.tabs.setMaximumWidth(465)
        layout.addWidget(self.tabs)
        self._shading_tab()
        self._model_tab()
        self._transforms_tab()
        self._camera_tab()
        self._menus()
        self.viewport.camera_changed.connect(self._camera_summary)
        self.set_scene([SceneObject(sphere())])
        self.last = time.monotonic()
        self.timer = C.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(70)
        self.statusBar().showMessage("ЛР 9  ·  Гуро / Ламберт  ·  Фонг / toon  ·  UV")

    @property
    def active(self):
        return self.items[self.selected]

    def _tab(self, title):
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        body = W.QWidget()
        body.setObjectName("tabbody")
        layout = W.QVBoxLayout(body)
        layout.setAlignment(C.Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(body)
        self.tabs.addTab(scroll,title)
        return layout

    @staticmethod
    def _group(parent,title):
        box = W.QGroupBox(title)
        layout = W.QVBoxLayout(box)
        parent.addWidget(box)
        return layout

    @staticmethod
    def _row(parent,label,*widgets):
        line = W.QHBoxLayout()
        if label:
            line.addWidget(W.QLabel(label))
        for widget in widgets:
            line.addWidget(widget,1)
        parent.addLayout(line)

    @staticmethod
    def _button(text, function, primary=False):
        button = W.QPushButton(text)
        if primary:
            button.setProperty("primary",True)
        button.clicked.connect(function)
        return button

    def _check(self,parent,title,checked,change):
        widget = W.QCheckBox(title)
        widget.setChecked(checked)
        widget.toggled.connect(change)
        parent.addWidget(widget)
        return widget

    def _shading_tab(self):
        layout = self._tab("ЛР 9")
        group = self._group(layout,"01  ·  Закраска модели")
        self.mode = W.QComboBox()
        for label,key in [("Гуро / Ламберт","gouraud"),("Фонг / тун-шейдинг","toon"),
                          ("Текстура UV","texture"),("Цвет граней (ЛР 8)","flat")]:
            self.mode.addItem(label,key)
        self.mode.currentIndexChanged.connect(self._change_mode)
        group.addWidget(self.mode)
        self.mode_tip = note("")
        group.addWidget(self.mode_tip)
        group = self._group(layout,"02  ·  Точечный источник света")
        self.light = [spin(v) for v in self.opts.light]
        self._row(group,"XYZ",*self.light)
        for box in self.light:
            box.valueChanged.connect(self._light_changed)
        self.color_button = self._button("Цвет объекта",self._choose_color)
        group.addWidget(self.color_button)
        self.bands = count(4,2,8)
        self.bands.valueChanged.connect(lambda n:self._option("levels",n))
        self._row(group,"Ступеней toon",self.bands)
        group = self._group(layout,"03  ·  Текстура и демонстрация")
        self._row(group,"",self._button("Открыть изображение…",self._load_texture,True),
                  self._button("UV-сетка",self._reset_texture))
        self.texture_label = note("Текстура: встроенная UV-сетка")
        group.addWidget(self.texture_label)
        self._row(group,"",
                  self._button("Тетраэдр",lambda:self._demo("Тетраэдр")),
                  self._button("Куб",lambda:self._demo("Гексаэдр (куб)")),
                  self._button("Октаэдр",lambda:self._demo("Октаэдр")))
        group.addWidget(self._button("Сфера для сравнения Гуро и toon",self._lighting_demo))
        self._change_mode()

    def _model_tab(self):
        layout = self._tab("Модели")
        group = self._group(layout,"Модели и OBJ")
        self.model_choice = W.QComboBox()
        self.model_choice.addItems([*SOLIDS,"Сфера"])
        self._row(group,"",self.model_choice,self._button("Показать",self._built_in))
        self._row(group,"",self._button("Открыть OBJ…",self._open),self._button("Экспорт OBJ…",self._save))
        group.addWidget(self._button("Сцена перекрытия из ЛР 8",self._overlap))
        self.object_choice = W.QComboBox()
        self.object_choice.currentIndexChanged.connect(self._select)
        self._row(group,"Объект",self.object_choice)
        self.model_info = note("")
        group.addWidget(self.model_info)
        group = self._group(layout,"Фигура вращения")
        group.addWidget(note("Образующая: одна точка X Y Z на строку."))
        self.profile = W.QPlainTextEdit(PROFILE_DEFAULT)
        self.profile.setFixedHeight(145)
        group.addWidget(self.profile)
        self.profile_axis = W.QComboBox()
        self.profile_axis.addItems(["Y","X","Z"])
        self.profile_steps = count(24,3,256)
        self._row(group,"Ось / N",self.profile_axis,self.profile_steps)
        self._row(group,"",self._button("Пример для оси",self._sample_profile),
                  self._button("Построить",self._make_revolution,True))
        group = self._group(layout,"Поверхность z = f(x, y)")
        preset = W.QComboBox()
        preset.addItems(FUNCTIONS)
        self.formula = W.QLineEdit(next(iter(FUNCTIONS.values())))
        preset.currentTextChanged.connect(lambda key:self.formula.setText(FUNCTIONS[key]))
        group.addWidget(preset)
        group.addWidget(self.formula)
        self.xr,self.yr = [spin(-3),spin(3)],[spin(-3),spin(3)]
        self.nxy = [count(24,2,150),count(24,2,150)]
        self._row(group,"X min / max",*self.xr)
        self._row(group,"Y min / max",*self.yr)
        self._row(group,"Nx / Ny",*self.nxy)
        group.addWidget(self._button("Построить поверхность", self._make_surface,True))

    def _transforms_tab(self):
        layout = self._tab("Матрицы")
        group = self._group(layout,"Перенос и масштаб XYZ")
        self.move = [spin() for _ in range(3)]
        self.resize_xyz = [spin(1,-100,100) for _ in range(3)]
        self._row(group,"Δ XYZ",*self.move)
        group.addWidget(self._button("Перенести",lambda:self._apply(translation(*values(self.move)))))
        self._row(group,"S XYZ",*self.resize_xyz)
        self._row(group,"",
                  self._button("От начала",lambda:self._apply(scaling(*values(self.resize_xyz)))),
                  self._button("От центра",lambda:self._center_apply(scaling(*values(self.resize_xyz)))))
        group = self._group(layout,"Поворот и отражение")
        self.rot_axis = W.QComboBox()
        self.rot_axis.addItems(["X","Y","Z"])
        self.rot_angle = spin(30,-3600,3600)
        self._row(group,"Ось / °",self.rot_axis,self.rot_angle)
        self._row(group,"",
                  self._button("Вокруг оси",lambda:self._apply(self._rotation())),
                  self._button("Через центр",lambda:self._center_apply(self._rotation())))
        self.plane = W.QComboBox()
        self.plane.addItems(["XY","XZ","YZ"])
        self._row(group,"Плоскость",self.plane,
                  self._button("Отразить",lambda:self._apply(reflection(self.plane.currentText()))))
        group = self._group(layout,"Вращение вокруг прямой AB")
        self.a,self.b = [spin(0) for _ in range(3)],[spin(1) for _ in range(3)]
        self.ab_angle = spin(45,-3600,3600)
        self._row(group,"A",*self.a)
        self._row(group,"B",*self.b)
        self._row(group,"Угол",self.ab_angle,self._button("Повернуть",self._rotate_ab))
        group = self._group(layout,"Матрица выбранной модели")
        self.matrix = note("")
        self.matrix.setFont(G.QFont("monospace",10))
        group.addWidget(self.matrix)
        self._row(group,"",self._button("Отменить",self._undo),self._button("Сбросить",self._reset_transform))

    def _camera_tab(self):
        layout = self._tab("Вид")
        group = self._group(layout,"Камера и проекция")
        self.projection = W.QComboBox()
        self.projection.addItem("Перспективная","perspective")
        self.projection.addItem("Аксонометрическая","orthographic")
        self.projection.currentIndexChanged.connect(self._projection)
        group.addWidget(self.projection)
        self.eye,self.direction = [spin(v) for v in (5,3,6)],[spin(v) for v in (-5,-3,-6)]
        self._row(group,"Позиция",*self.eye)
        self._row(group,"Вектор",*self.direction)
        self._row(group,"",self._button("Установить камеру",self._set_camera),
                  self._button("Вписать сцену",self.viewport.fit,True))
        fov = spin(48,20,120)
        fov.valueChanged.connect(self._set_fov)
        self._row(group,"Угол обзора °",fov)
        self.camera_info = note("")
        group.addWidget(self.camera_info)
        group = self._group(layout,"Видимость и диагностика ЛР 8")
        for label,key in [("Отсекать нелицевые грани","cull_backfaces"),("Использовать Z-буфер","use_zbuffer"),
                          ("Закрашивать грани","draw_faces"),("Показать рёбра","draw_edges"),
                          ("Показать нормали граней","draw_normals")]:
            self._check(group,label,getattr(self.opts,key),lambda val,k=key:self._option(k,val))
        self._check(group,"Показать оси",False,self._axes)
        group = self._group(layout,"Анимация")
        self.orbit = self._check(group,"Орбита камеры (перспектива)",False,self._orbit_toggled)
        self.spin_object = self._check(group,"Вращать объект вокруг Y",False,self._spin_toggled)
        self.speed = spin(35,-180,180)
        self._row(group,"Скорость °/с",self.speed)

    def _menus(self):
        menu = self.menuBar().addMenu("Файл")
        for name,shortcut,action in [
            ("Открыть OBJ…","Ctrl+O",self._open),("Экспорт OBJ…","Ctrl+E",self._save),
            ("Сохранить вид PNG…","Ctrl+S",self._screenshot),("Отменить преобразование","Ctrl+Z",self._undo)]:
            item = menu.addAction(name)
            item.setShortcut(shortcut)
            item.triggered.connect(action)

    def _option(self,key,value):
        setattr(self.opts,key,value)
        self.viewport.update()

    def _change_mode(self):
        mode = self.mode.currentData()
        self._option("mode",mode)
        self.mode_tip.setText({
            "gouraud":"Ламберт вычисляется в вершинах. Между ними интерполируется цвет.",
            "toon":"В каждом пикселе нормаль интерполируется, нормализуется и даёт одну из ступеней цвета.",
            "texture":"UV интерполируются с поправкой на перспективу. RGB берётся из изображения.",
            "flat":"Постоянный цвет на грань для сравнения отсечения и Z-буфера."
        }[mode])

    def _light_changed(self):
        self._option("light",tuple(values(self.light)))

    def _choose_color(self):
        q = W.QColorDialog.getColor(G.QColor(*self.active.color),self,"Цвет объекта")
        if q.isValid():
            self.active.color = (q.red(),q.green(),q.blue())
            self._refresh()

    def _load_texture(self):
        filename,_ = W.QFileDialog.getOpenFileName(self,"Открыть текстуру",str(ROOT/"examples"),
                                                  "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not filename:
            return
        image = G.QImage(filename)
        if image.isNull():
            self._error("Текстура","Не удалось прочитать изображение.")
            return
        if max(image.width(),image.height())>2048:
            image = image.scaled(2048,2048,C.Qt.AspectRatioMode.KeepAspectRatio,
                                 C.Qt.TransformationMode.SmoothTransformation)
        image = image.convertToFormat(G.QImage.Format.Format_RGBA8888)
        pointer = image.constBits()
        if hasattr(pointer,"setsize"):
            pointer.setsize(image.sizeInBytes())
        data = np.frombuffer(pointer,dtype=np.uint8).reshape(image.height(),image.bytesPerLine())
        array = data[:,:image.width()*4].reshape(image.height(),image.width(),4)[:,:,:3].copy()
        self._option("texture",array)
        self.mode.setCurrentIndex(self.mode.findData("texture"))
        self.texture_label.setText(f"Текстура: {Path(filename).name}")

    def _reset_texture(self):
        self._option("texture",None)
        self.texture_label.setText("Текстура: встроенная UV-сетка")
        self.mode.setCurrentIndex(self.mode.findData("texture"))

    def set_scene(self,items):
        self.items,self.selected,self.histories = list(items),0,{}
        self.viewport.items = self.items
        self.object_choice.blockSignals(True)
        self.object_choice.clear()
        self.object_choice.addItems([o.name for o in items])
        self.object_choice.blockSignals(False)
        self.viewport.fit()
        self._refresh()

    def _select(self,index):
        if index >= 0:
            self.selected = index
            self._refresh()

    def _refresh(self):
        model = self.active.model
        self.heading.setText(self.active.name)
        self.model_info.setText(f"{len(model.vertices)} вершин, {len(model.faces)} граней, {len(model.texcoords)} UV")
        self.matrix.setText("\n".join("  ".join(f"{v:7.3f}" for v in row) for row in model.model_matrix))
        self.color_button.setText("Цвет объекта  ·  "+G.QColor(*self.active.color).name())
        self.viewport.update()

    def _demo(self,name):
        self.mode.setCurrentIndex(self.mode.findData("texture"))
        self.set_scene([SceneObject(SOLIDS[name]())])

    def _lighting_demo(self):
        if self.mode.currentData() not in ("gouraud","toon"):
            self.mode.setCurrentIndex(self.mode.findData("gouraud"))
        self.set_scene([SceneObject(sphere())])

    def _built_in(self):
        name = self.model_choice.currentText()
        self.set_scene([SceneObject(sphere() if name=="Сфера" else SOLIDS[name]())])

    def _overlap(self):
        self.mode.setCurrentIndex(self.mode.findData("flat"))
        self.set_scene(overlap_scene())

    def _open(self):
        filename,_ = W.QFileDialog.getOpenFileName(self,"Открыть OBJ",str(ROOT/"examples"),"Wavefront OBJ (*.obj)")
        if filename:
            try:
                self.set_scene([SceneObject(load_obj(filename))])
            except (ValueError,OSError) as err:
                self._error("Импорт OBJ",err)

    def _save(self):
        filename,_ = W.QFileDialog.getSaveFileName(self,"Сохранить выбранную модель","model.obj","Wavefront OBJ (*.obj)")
        if filename:
            try:
                path = Path(filename)
                save_obj(self.active.model,path if path.suffix else path.with_suffix(".obj"))
            except (ValueError,OSError) as err:
                self._error("Экспорт OBJ",err)

    def _screenshot(self):
        file,_ = W.QFileDialog.getSaveFileName(self,"Сохранить вид","lab9.png","Изображение PNG (*.png)")
        if file and not self.viewport.grab().save(file,"PNG"):
            self._error("Сохранение PNG","Не удалось записать изображение.")

    def _sample_profile(self):
        axis = self.profile_axis.currentText()
        order = {"Y":(0,1,2),"X":(1,0,2),"Z":(0,2,1)}[axis]
        self.profile.setPlainText("\n".join(" ".join(f"{p.as_array()[i]:g}" for i in order)
                                            for p in read_profile(PROFILE_DEFAULT)))

    def _make_revolution(self):
        try:
            mesh = revolution(read_profile(self.profile.toPlainText()),self.profile_axis.currentText(),self.profile_steps.value())
            self.set_scene([SceneObject(mesh)])
        except ValueError as err:
            self._error("Фигура вращения",err)

    def _make_surface(self):
        try:
            mesh = function_surface(self.formula.text(),*values(self.xr),*values(self.yr),*values(self.nxy))
            self.set_scene([SceneObject(mesh)])
        except ValueError as err:
            self._error("Поверхность",err)

    def _apply(self,matrix):
        model = self.active.model
        before = model.model_matrix.copy()
        try:
            model.apply(matrix)
        except ValueError as err:
            self._error("Преобразование",err)
            return
        self.histories.setdefault(self.selected,[]).append(before)
        self._refresh()

    def _center_apply(self,matrix):
        self._apply(around_point(matrix,self.active.model.center))

    def _rotation(self):
        return rotation_axis(self.rot_axis.currentText(),self.rot_angle.value())

    def _rotate_ab(self):
        try:
            self._apply(rotate_around_line(values(self.a),values(self.b),self.ab_angle.value()))
        except ValueError as err:
            self._error("Вращение",err)

    def _undo(self):
        past = self.histories.get(self.selected)
        if past:
            self.active.model.model_matrix = past.pop()
            self._refresh()

    def _reset_transform(self):
        self.histories.setdefault(self.selected,[]).append(self.active.model.model_matrix.copy())
        self.active.model.reset()
        self._refresh()

    def _projection(self):
        self.viewport.camera.projection = self.projection.currentData()
        if self.orbit.isChecked() and self.projection.currentIndex()!=0:
            self.orbit.setChecked(False)
        self.viewport.update()

    def _set_camera(self):
        try:
            self.viewport.camera.set_pose(values(self.eye),values(self.direction))
            self._camera_summary()
            self.viewport.update()
        except ValueError as err:
            self._error("Камера",err)

    def _camera_summary(self):
        cam = self.viewport.camera
        for box,value in zip(self.eye+self.direction,[*cam.position,*cam.direction]):
            box.setValue(float(value))
        self.camera_info.setText("Цель орбиты: "+", ".join(f"{v:.2f}" for v in cam.target))

    def _set_fov(self,value):
        self.viewport.camera.fov_degrees = value
        self.viewport.update()

    def _axes(self,on):
        self.viewport.show_axes = on
        self.viewport.update()

    def _orbit_toggled(self,on):
        if on:
            self.spin_object.setChecked(False)
            self.projection.setCurrentIndex(0)

    def _spin_toggled(self,on):
        if on:
            self.orbit.setChecked(False)

    def _tick(self):
        now = time.monotonic()
        dt,self.last = min(.25,now-self.last),now
        if self.orbit.isChecked() and self.speed.value():
            self.viewport.camera.orbit(self.speed.value()*dt)
            self._camera_summary()
            self.viewport.update()
        elif self.spin_object.isChecked() and self.speed.value():
            mesh = self.active.model
            mesh.apply(around_point(rotation_axis("Y",self.speed.value()*dt),mesh.center))
            self._refresh()

    def _error(self,title,err):
        W.QMessageBox.warning(self,title,str(err))
