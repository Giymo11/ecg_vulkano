use glm::*;

const SENSITIVITY: f32 = 0.005;

pub struct ArcballCamera {
    mouse_dragging: bool,
    sensitivity: f32,
    last_position: (f64, f64),
    r: f32,
    theta: f32,
    phi: f32,
}

use winit::{WindowEvent, MouseButton, ElementState, MouseScrollDelta};
use crate::Controller;

impl ArcballCamera {
    pub fn new(r: f32, theta: f32, phi: f32) -> Self {
        ArcballCamera {
            mouse_dragging: false,
            sensitivity: SENSITIVITY,
            last_position: (0.0, 0.0),
            r,
            theta,
            phi,
        }
    }
    fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left && state == ElementState::Pressed {
            self.mouse_dragging = true;
        } else if button == MouseButton::Left && state == ElementState::Released {
            self.mouse_dragging = false;
        }
    }
    fn cursor_input(&mut self, new_position: winit::dpi::LogicalPosition) {
        if self.last_position.0 < 0.1 {
            self.last_position = new_position.into();
        } else {
            let x_offset = (new_position.x - self.last_position.0) as f32;
            let y_offset = (self.last_position.1 - new_position.y) as f32;
            self.last_position = new_position.into();

            if self.mouse_dragging {
                self.phi -= x_offset * self.sensitivity;
                self.theta += y_offset * self.sensitivity;
                self.theta = na::clamp(self.theta, 0.1f32.to_radians(), 179.9f32.to_radians());
            }
        }
    }
    fn mousewheel_input(&mut self, delta_y: f32) {
        self.r -= delta_y;
        self.r = na::clamp(self.r, 0.1, 200.0);
    }

}

impl Controller for ArcballCamera {
    fn use_window_event(&mut self, ev: &WindowEvent) -> bool {
        match ev {
            WindowEvent::MouseInput {
                device_id: _,
                state,
                button,
                ..
            } => self.mouse_input(*button, *state),

            WindowEvent::CursorMoved {
                device_id: _,
                position: new_position,
                ..
            } => self.cursor_input(*new_position),

            WindowEvent::MouseWheel {
                device_id: _,
                delta: MouseScrollDelta::LineDelta(_, delta_y),
                ..
            } => self.mousewheel_input(*delta_y),

            _ => return false,
        }
        true
    }

    fn generate_view_mat(&self) -> Mat4x4 {
        let x = self.r * self.theta.sin() * self.phi.cos();
        let y = self.r * self.theta.sin() * self.phi.sin();
        let z = self.r * self.theta.cos();

        let trans = translation(&vec3(y, z, x));
        let roty = rotation(self.phi, &vec3(0., 1., 0.));
        let rotx = rotation(self.theta - 90.0f32.to_radians(), &vec3(1., 0., 0.));

        let camera_mat = trans * roty * rotx;
        inverse(&camera_mat)
    }
}