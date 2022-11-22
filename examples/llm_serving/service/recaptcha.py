"""
Adapted from https://github.com/mardix/flask-recaptcha

The new Google ReCaptcha implementation for Flask without Flask-WTF
Can be used as standalone
"""

__NAME__ = "Flask-ReCaptcha"
__version__ = "0.5.0"
__license__ = "MIT"
__author__ = "Mardix"
__copyright__ = "(c) 2015 Mardix"

import json

#from flask import request
try:
    from jinja2 import Markup
except ImportError:
    from jinja2.utils import markupsafe
    Markup = markupsafe.Markup
import requests

from llm_serving.service.constants import USE_RECAPTCHA, KEYS_FILENAME


class DEFAULTS(object):
    IS_ENABLED = True
    THEME = "light"
    TYPE = "image"
    SIZE = "normal"
    LANGUAGE = "en"
    TABINDEX = 0


class ReCaptcha(object):

    VERIFY_URL = "https://www.recaptcha.net/recaptcha/api/siteverify"

    def __init__(self, app=None, site_key=None, secret_key=None, is_enabled=True, **kwargs):
        if app:
            self.init_app(app=app)
        else:
            self.site_key = site_key
            self.secret_key = secret_key
            self.is_enabled = is_enabled
            self.theme = kwargs.get('theme', DEFAULTS.THEME)
            self.type = kwargs.get('type', DEFAULTS.TYPE)
            self.size = kwargs.get('size', DEFAULTS.SIZE)
            self.language = kwargs.get('language', DEFAULTS.LANGUAGE)
            self.tabindex = kwargs.get('tabindex', DEFAULTS.TABINDEX)

    def init_app(self, app=None):
        self.__init__(site_key=app.config.get("RECAPTCHA_SITE_KEY"),
                      secret_key=app.config.get("RECAPTCHA_SECRET_KEY"),
                      is_enabled=app.config.get("RECAPTCHA_ENABLED", DEFAULTS.IS_ENABLED),
                      theme=app.config.get("RECAPTCHA_THEME", DEFAULTS.THEME),
                      type=app.config.get("RECAPTCHA_TYPE", DEFAULTS.TYPE),
                      size=app.config.get("RECAPTCHA_SIZE", DEFAULTS.SIZE),
                      language=app.config.get("RECAPTCHA_LANGUAGE", DEFAULTS.LANGUAGE),
                      tabindex=app.config.get("RECAPTCHA_TABINDEX", DEFAULTS.TABINDEX))

        @app.context_processor
        def get_code():
            return dict(recaptcha=self.get_code())

    def get_code(self):
        """
        Returns the new ReCaptcha code
        :return:
        """
        raw = "" if not self.is_enabled else ("""
        <script src='//www.recaptcha.net/recaptcha/api.js?hl={LANGUAGE}'></script>
        <div class="g-recaptcha" data-sitekey="{SITE_KEY}" data-theme="{THEME}" data-type="{TYPE}" data-size="{SIZE}"\
         data-tabindex="{TABINDEX}"></div>
        """.format(SITE_KEY=self.site_key, THEME=self.theme, TYPE=self.type, SIZE=self.size, LANGUAGE=self.language, TABINDEX=self.tabindex))
        return Markup(raw)

    def verify(self, response=None, remote_ip=None):
        if self.is_enabled:
            data = {
                "secret": self.secret_key,
                "response": response,# or request.json.get('g-recaptcha-response', ""),
                "remoteip": remote_ip,# or request.environ.get('REMOTE_ADDR')
            }

            r = requests.get(self.VERIFY_URL, params=data)
            return r.json()["success"] if r.status_code == 200 else False
        return True


def load_recaptcha(use_recaptcha):
    if use_recaptcha:
        keys = json.load(open(KEYS_FILENAME, "r"))
        recaptcha = ReCaptcha(site_key=keys["RECAPTCHA_SITE_KEY"],
                              secret_key=keys["RECAPTCHA_SECRET_KEY"])
    else:
        recaptcha = ReCaptcha(is_enabled=False)
    return recaptcha
