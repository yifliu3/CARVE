"""Offline/online proxy rendering through the same browser camera and shader."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image


class BrowserRenderer:
    def __init__(self, base_url, executable=None):
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        self.browser = None
        try:
            self.browser = self.playwright.chromium.launch(
                headless=True, executable_path=executable or None,
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
            self.page = self.browser.new_page()
            self.page.goto(base_url.rstrip("/") + "/render.html")
            self.page.wait_for_function("window.ready === true")
        except BaseException:
            self.close()
            raise
        self.scene = None

    def render(self, scene, poses):
        if scene != self.scene:
            self.page.evaluate("url => window.renderer.load(url)", f"/scenes/{scene}/environment")
            self.scene = scene
        frames = []
        for pose in poses:
            encoded = self.page.evaluate("pose => window.renderer.png(pose)", pose)
            with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                frames.append(np.asarray(image.convert("RGB")))
        return np.stack(frames)

    def close(self):
        if self.browser is not None:
            self.browser.close()
        self.playwright.stop()
