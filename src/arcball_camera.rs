

use cgmath::prelude::*;
use cgmath::{Matrix4, Vector3, Vector4, Rad, Deg};

const SENSITIVITY: f32 = 0.005;

pub struct ArcballCamera {
    mouse_dragging: bool,
    sensitivity: f32,
    last_position: (f64, f64),
    r: f32,
    theta: Deg<f32>,
    phi: f32,
}

use winit::{WindowEvent, MouseButton, ElementState, MouseScrollDelta};

impl ArcballCamera {
    pub fn new(r: f32, theta: Deg<f32>, phi: f32) -> Self {
        ArcballCamera {
            mouse_dragging: false,
            sensitivity: SENSITIVITY,
            last_position: (0.0, 0.0),
            r,
            theta,
            phi,
        }
    }
    pub fn use_window_event(&mut self, ev: &WindowEvent) -> bool {
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
                self.theta += Rad(y_offset * self.sensitivity).into();
                self.theta = na::clamp(self.theta, Deg(0.1), Deg(179.9));
            }
        }
    }
    fn mousewheel_input(&mut self, delta_y: f32) {
        self.r -= delta_y;
        self.r = na::clamp(self.r, 0.1, 200.0);
    }

    pub fn generate_view_mat(&self) -> Matrix4<f32>{
        let x = self.r * self.theta.sin() * self.phi.cos();
        let y = self.r * self.theta.sin() * self.phi.sin();
        let z = self.r * self.theta.cos();

        let trans = Matrix4::from_translation(Vector3::new(y, z, x));
        let roty = Matrix4::from_angle_y(Rad(self.phi));
        let rotx = Matrix4::from_angle_x(self.theta - Deg(90.0));

        (trans * roty * rotx).invert().unwrap()
    }
}