#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;
extern crate winit;

use vulkano_win::VkSurfaceBuild;
use winit::EventsLoop;
use winit::WindowBuilder;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

use config::Config;

#[cfg(debug)]
fn get_extensions() {
    retuInstanceExtensions {
        ext_debug_report: true,
        ..vulkano_win::required_extensions()
    };
}

fn main() {
    println!("Hello, world!");

    let mut settings = Config::new();
    settings
        .merge(config::File::with_name("assets/settings"))
        .expect("Failed to read settings file!");
    let settings_width = settings.get::<u32>("window.width").unwrap_or(800);
    let settings_height = settings.get::<u32>("window.height").unwrap_or(600);
    let settings_title = settings
        .get_str("window.title")
        .unwrap_or("Please enter a window title!".to_string());

    // get vulkan instance
    #[cfg(debug_assertions)]
    let instance = {
        use vulkano::instance::InstanceExtensions;

        let extensions = InstanceExtensions {
            ext_debug_report: true,
            ..vulkano_win::required_extensions()
        };
        // NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers list:
        let layer = "VK_LAYER_LUNARG_standard_validation";
        let layers = vec![layer];
        Instance::new(None, &extensions, layers).expect("failed to create Vulkan instance")
    };

    #[cfg(not(debug_assertions))]
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };

    #[cfg(debug_assertions)]
    let _debug_callback = {
        use vulkano::instance::debug::DebugCallback;
        use vulkano::instance::debug::MessageTypes;
        // create and use the debug callbacks
        let all_messages = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: true,
            debug: true,
        };
        DebugCallback::new(&instance, all_messages, |msg| {
            let ty = if msg.ty.error {
                "error"
            } else if msg.ty.warning {
                "warning"
            } else if msg.ty.performance_warning {
                "performance_warning"
            } else if msg.ty.information {
                "information"
            } else if msg.ty.debug {
                "debug"
            } else {
                panic!("no-impl");
            };
            println!("{} {}: {}", msg.layer_prefix, ty, msg.description);
        })
    };

    // choose physical device
    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    // build window and surface
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .with_title(settings_title)
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();

    // We take the first queue that supports drawing to our window.
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphical queue family");

    // create the logical device
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..vulkano::device::DeviceExtensions::none()
        };

        Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    // the dimensions of the viewport, mutable as it can be resized
    let mut dimensions;

    let (mut swapchain, mut images) = {
        let caps = surface
            .capabilities(physical)
            .expect("failed to get surface capabilities");

        dimensions = caps
            .current_extent
            .unwrap_or([settings_width, settings_height]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            caps.supported_usage_flags,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo, // TODO: check if a better one is supported
            true,
            None,
        )
        .expect("failed to create swapchain")
    };

    // create a buffer that will store the shape of our triangle
    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            [
                Vertex {
                    position: [-0.5, -0.25],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.25, -0.1],
                },
            ]
                .iter()
                .cloned(),
        )
        .expect("failed to create buffer")
    };

    // create the vertex and fragment shaders
    mod vs {
        #[derive(VulkanoShader)]
        #[ty = "vertex"]
        #[src = "
#version 450
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"]
        #[allow(dead_code)]
        struct Dummy;
    }

    mod fs {
        #[derive(VulkanoShader)]
        #[ty = "fragment"]
        #[src = "
#version 450
layout(location = 0) out vec4 f_color;
void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"]
        #[allow(dead_code)]
        struct Dummy;
    }

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    // the *render pass* describes where the output of the graphics pipeline will go
    let render_pass = Arc::new(
        single_pass_renderpass!(device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            depth_stencil: {}
        }
    )
        .unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            // We need to indicate the layout of the vertices.
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            // The content of the vertex buffer describes a list of triangles.
            .triangle_list()
            // Use a resizable viewport set to draw over the entire window
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            // We have to indicate in which render pass the pipeline should be used.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    // Before we can draw we also need to create the actual framebuffers.
    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_, _>>>> = None;

    /* -----------------------------------------
    // Initialization finished!
    -------------------------------------------- */

    // to track if we need to recreate the swapchain after a resize
    let mut recreate_swapchain = false;

    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        }]),
        scissors: None,
    };

    loop {
        previous_frame_end.cleanup_finished();

        // If the swapchain needs to be recreated, recreate it
        if recreate_swapchain {
            // Get the new dimensions for the viewport/framebuffers.
            dimensions = surface
                .capabilities(physical)
                .expect("failed to get surface capabilities")
                .current_extent
                .unwrap();

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

            swapchain = new_swapchain;
            images = new_images;

            framebuffers = None;

            dynamic_state.viewports = Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            }]);

            recreate_swapchain = false;
        }

        // Because framebuffers contains an Arc on the old swapchain, we need to
        // recreate framebuffers as well.
        if framebuffers.is_none() {
            framebuffers = Some(
                images
                    .iter()
                    .map(|image| {
                        Arc::new(
                            Framebuffer::start(render_pass.clone())
                                .add(image.clone())
                                .unwrap()
                                .build()
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain.
        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        // The command buffer object holds the list of commands that are going to be executed.
        let command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(
                    framebuffers.as_ref().unwrap()[image_num].clone(),
                    false,
                    vec![[1.0, 1.0, 1.0, 1.0].into()],
                )
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();

        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time. This happens when the
        // GPU's queue is full and the driver has to wait until the GPU finished some work.
        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            winit::Event::DeviceEvent {
                event: winit::DeviceEvent::Key(pressed_key),
                ..
            } => {
                if pressed_key.state == winit::ElementState::Released && pressed_key.virtual_keycode == Some(winit::VirtualKeyCode::Escape) {
                    done = true
                } else {
                    println!("Key Pressed:  {:?}", pressed_key)
                }
            }
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => done = true,
            _ => (),
        });
        if done {
            return;
        }
    }
}
