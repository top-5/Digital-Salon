#include "EglSmoke.h"
#ifdef __linux__
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <iostream>

int RunEglSmoke()
{
    auto query = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto platform = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLint major = 0, minor = 0;
    if (query && platform) {
        EGLDeviceEXT devices[16];
        EGLint count = 0;
        if (query(16, devices, &count)) {
            for (int i = 0; i < count; ++i) {
                auto candidate = platform(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                if (candidate != EGL_NO_DISPLAY && eglInitialize(candidate, &major, &minor)) {
                    display = candidate;
                    break;
                }
            }
        }
    }
    if (display == EGL_NO_DISPLAY) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
            std::cerr << "Cannot initialize headless EGL: " << eglGetError() << '\n';
            return 1;
        }
    }
    const EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE};
    EGLConfig config;
    EGLint count = 0;
    if (!eglBindAPI(EGL_OPENGL_API) || !eglChooseConfig(display, attributes, &config, 1, &count) || !count) {
        std::cerr << "No EGL OpenGL pbuffer configuration\n";
        eglTerminate(display);
        return 1;
    }
    const EGLint dimensions[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
    auto surface = eglCreatePbufferSurface(display, config, dimensions);
    auto context = eglCreateContext(display, config, EGL_NO_CONTEXT, nullptr);
    int result = 1;
    if (surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(display, surface, surface, context)) {
        glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        unsigned char pixel[4] = {};
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
        result = glGetError() == GL_NO_ERROR && pixel[1] >= 126 && pixel[1] <= 129 ? 0 : 1;
        std::cout << "EGL " << major << '.' << minor << "; OpenGL " << glGetString(GL_VERSION)
                  << "; renderer " << glGetString(GL_RENDERER) << "; pbuffer readback "
                  << (result == 0 ? "passed" : "failed") << '\n';
    } else {
        std::cerr << "Cannot create/make current EGL context: " << eglGetError() << '\n';
    }
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
    if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
    eglTerminate(display);
    return result;
}
#endif
