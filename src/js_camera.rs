use crate::Controller;




use deno::*;
use futures::future::lazy;

use tokio::prelude::*;
use deno::*;


use winit::*;
use glm::*;
use std::sync::Arc;

pub struct JsCamera {

}

pub fn dispatch_fn(control: &[u8], zero_copy_buf: Option<PinnedBuf>) -> Op {
    println!("dispatch got back: {}", control[0]);
    let buf = vec![43u8].into_boxed_slice();
    Op::Sync(buf)
}

impl JsCamera {
    pub fn new() -> JsCamera {
        let startup_data = StartupData::Script(Script {
            source: r#"a = 12;


        let response = Deno.core.dispatch(new Uint8Array([42]));
        Deno.core.dispatch(response);
        "#,
            filename: "sause.js",
        });

        let mut config = Config::default();
        config.dispatch(dispatch_fn);

        let mut isolate = deno::Isolate::new(startup_data, config);
        println!("started isolate");

        JsCamera{}
    }
}

impl Controller for JsCamera {

    fn use_window_event(&mut self, ev: &WindowEvent) -> bool {
        false
    }

    fn generate_view_mat(&self) -> Mat4x4 {
        inverse(&translation(&vec3(0., 0., 1.)))
    }
}