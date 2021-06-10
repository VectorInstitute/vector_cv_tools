from anomaly_detection import anomaly_detection_page
from covid_video_classification import video_classification_page
from system_level_diagram import system_level_diagram

_DEMO_REGISTRY = {}


def register_demo(name):

    def _decorator(f):
        if name in _DEMO_REGISTRY:
            raise ValueError(
                "Demo name {} already registered for function{}".format(
                    name, _DEMO_REGISTRY[name]))

        _DEMO_REGISTRY[name] = f

        return f

    return _decorator


def get_all_demos():
    return list(_DEMO_REGISTRY.keys())


def get_demo(name):
    return _DEMO_REGISTRY[name]


@register_demo("Anomaly Detection")
def __page(*args, **kwargs):
    anomaly_detection_page(*args, **kwargs)


@register_demo("COVID Ultra-sound Video Classification")
def __page(*args, **kwargs):
    video_classification_page(*args, **kwargs)


@register_demo("System Level Diagram")
def __page(*args, **kwargs):
    system_level_diagram(*args, **kwargs)
