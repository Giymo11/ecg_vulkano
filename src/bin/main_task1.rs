
// This is a triangle example!
// heavily copied from https://github.com/vulkano-rs/vulkano-examples/blob/master/src/bin/triangle.rs


// The `vulkano` crate is the main crate that you must use to use Vulkan.
#[macro_use]
extern crate vulkano;
// Provides the `shader!` macro that is used to generate code for using shaders.
extern crate vulkano_shaders;
// The Vulkan library doesn't provide any functionality to create and handle windows.
extern crate winit;
// The `vulkano_win` crate is the link between `vulkano` and `winit`.
extern crate vulkano_win;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder, Event, WindowEvent};
use winit::dpi::LogicalSize;

use std::sync::Arc;
use config::Config;


#[cfg(debug_assertions)]
fn create_instance() -> Arc<Instance> {
    use vulkano::instance::InstanceExtensions;

    let extensions = InstanceExtensions {
        ext_debug_report: true,
        ..vulkano_win::required_extensions()
    };
    // NOTE: To simplify the example code we won't verify these layer(s) are actually in the layers list:
    let layer = "VK_LAYER_LUNARG_standard_validation";
    let layers = vec![layer];
    Instance::new(None, &extensions, layers).expect("failed to create Vulkan instance")
}

#[cfg(not(debug_assertions))]
fn create_instance() -> Arc<Instance> {
    let extensions = vulkano_win::required_extensions();
    Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
}


fn main() {
    // Uses config-rs to parse a .toml file at assets/settings.toml
    let mut settings = Config::new();
    settings
        .merge(config::File::with_name("assets/settings"))
        .expect("Failed to read settings file!");
    let settings_width = settings.get::<u32>("window.width").unwrap_or(800);
    let settings_height = settings.get::<u32>("window.height").unwrap_or(600);
    let settings_title = settings
        .get_str("window.title")
        .unwrap_or("Please enter a window title!".to_string());

    // TODO: Debugging
    // TODO: ESC callback

    // The first step of any Vulkan program is to create an instance.
    let instance = create_instance();

    // If we let this debug_callback binding fall out of scope then it will stop providing events
    #[cfg(debug_assertions)]
    let _debug_callback = {
        use vulkano::instance::debug::DebugCallback;
        use vulkano::instance::debug::MessageTypes;
        // create and use the debug callbacks
        let all_messages = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: false,
            debug: false,
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

    // TODO: check for best option of physical device
    // For the sake of the example we are just going to use the first device.
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    // Some little debug infos.
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());


    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .with_dimensions(LogicalSize::from((settings_width, settings_height)))
        .with_title(settings_title)
        .build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

    // The next step is to choose which GPU queue will execute our draw commands.
    //
    // In a real-life application, we would probably use at least a graphics queue and a transfers
    // queue to handle data transfers in parallel. In this example we only use one queue.
    let queue_family = physical.queue_families().find(|&q| {
        // We take the first queue that supports drawing to our window.
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();


    // We have to pass five parameters when creating a device:
    // - Which physical device to connect to.
    // - A list of optional features and extensions that our program needs to work correctly.
    // - A list of layers to enable. This is very niche, and you will usually pass `None`.
    // - The list of queues that we are going to use.
    // The list of created queues is returned by the function alongside with the device.
    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned()
    ).unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator.
    let queue = queues.next().unwrap();

    // Before we can draw on the surface, we have to create what is called a swapchain.
    let (mut swapchain, images) = {
        // Querying the capabilities of the surface.
        let caps = surface.capabilities(physical).unwrap();

        let usage = caps.supported_usage_flags;

        // The alpha mode indicates how the alpha value of the final image will behave.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // The dimensions of the window, only used to initially setup the swapchain.
        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            // convert to physical pixels
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            // The window no longer exists so exit the application.
            return;
        };

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            initial_dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            true,
            None
        ).unwrap()

    };

    // We now create a buffer that will store the shape of our triangle.
    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { position: [-0.5, -0.25] },
            Vertex { position: [0.0, 0.5] },
            Vertex { position: [0.25, -0.1] }
        ].iter().cloned()).unwrap()
    };

    // The next step is to create the shaders.
    //
    // The raw shader creation API provided by the vulkano library is unsafe, for various reasons.
    //
    // An overview of what the `vulkano_shaders::shader!` macro generates can be found in the
    // `vulkano-shaders` crate docs. You can view them at https://docs.rs/vulkano-shaders/
    //
    // TODO: explain this in details
    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
#version 450
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
        }
    }

    mod fs {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: "
#version 450
layout(location = 0) out vec4 f_color;
void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // A render pass is an object that describes where the output of the graphics pipeline will go.
    let render_pass = Arc::new(single_pass_renderpass!(
        device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // We ask the GPU to clear the content of this attachment at the start of the drawing.
                load: Clear,
                // Store the output of the draw in the actual image. Alternative is to discard.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image.
                format: swapchain.format(),
                // TODO: explain what this is all about
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            // No depth-stencil attachment is indicated with empty brackets.
            depth_stencil: {}
        }
    ).unwrap());

    // Before we draw we have to create what is called a pipeline.
    let pipeline = Arc::new(GraphicsPipeline::start()
        // We need to indicate the layout of the vertices.
        .vertex_input_single_buffer()
        // A Vulkan shader can in theory contain multiple entry points, so we have to specify.
        .vertex_shader(vs.main_entry_point(), ())
        // The content of the vertex buffer describes a list of triangles.
        .triangle_list()
        // Use a resizable viewport set to draw over the entire window
        .viewports_dynamic_scissors_irrelevant(1)
        // See `vertex_shader`.
        .fragment_shader(fs.main_entry_point(), ())
        // The pipeline will only be usable from this particular subpass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
        .build(device.clone())
        .unwrap());

    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };

    // The render pass we created above only describes the layout of our framebuffers.
    let mut framebuffers = window_size_dependent_setup(
        &images,
        render_pass.clone(),
        &mut dynamic_state
    );

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<GpuFuture>;

    loop {
        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        previous_frame_end.cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        if recreate_swapchain {
            // Get the new dimensions of the window.
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_swapchain_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            framebuffers = window_size_dependent_setup(&new_swapchain_images, render_pass.clone(), &mut dynamic_state);

            recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain.
        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec!([0.0, 0.0, 1.0, 1.0].into());

        // In order to draw, we have to build a *command buffer*.
        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family()
        ).unwrap()
            // Before we can draw, we have to *enter a render pass*.
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()

            // We are now inside the first subpass of the render pass. We add a draw command.
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
            .unwrap()

            // We leave the render pass by calling `draw_end`.
            .end_render_pass()
            .unwrap()

            // Finish building the command buffer by calling `build`.
            .build().unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            // The color output is now expected to contain our triangle.
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        // Note that in more complex programs it is likely that one of `acquire_next_image`,
        // `command_buffer::submit`, or `present` will block for some time.
        // Blocking may be the desired behavior, but if you don't want to block you should spawn a
        // separate thread dedicated to submissions.

        // Handling the window events in order to close the program when the user wants to close
        // it.
        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(_) => recreate_swapchain = true,
                    WindowEvent::CloseRequested => done = true,
                    WindowEvent::KeyboardInput {
                        input:
                        winit::KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state,
                            ..
                        },
                        ..
                    } => match (virtual_code, state) {
                        (winit::VirtualKeyCode::Escape, _) => done = true,
                        _ => (),
                    },
                    _ => (),
                },
                _ => (),
            }
        });
        if done { return; }
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}