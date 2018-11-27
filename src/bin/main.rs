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
extern crate nalgebra as na;

use std::iter;
use std::sync::Arc;
use std::time::Instant;

use config::Config;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::debug::{DebugCallback, DebugCallbackCreationError};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use winit::{EventsLoop, Window, WindowBuilder, Event, WindowEvent};
use winit::dpi::LogicalSize;

use vulkano_win::VkSurfaceBuild;

use cgmath::{Matrix4, Vector3, Vector4, Deg};

use ecg_vulkano::*;


mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            path: "assets/simple_vert.glsl"
        }
}

mod fs {
    vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/simple_frag.glsl"
        }
}

// ------------------------------------------------------------------------------------------------
// TODO: swap all cgmath for nalgebra!
// ------------------------------------------------------------------------------------------------


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

#[cfg(debug_assertions)]
fn create_debug_callback(instance: Arc<Instance>) -> Result<DebugCallback, DebugCallbackCreationError> {
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
}

fn create_window(instance: Arc<Instance>, settings_width: u32, settings_height: u32, settings_title: String,
) -> Result<(EventsLoop, Arc<Surface<Window>>, [u32; 2]), String> {
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .with_dimensions(LogicalSize::from((settings_width, settings_height)))
        .with_title(settings_title)
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    // The dimensions of the window, only used to initially setup the swapchain.
    let dimensions = if let Some(dimensions) = window.get_inner_size() {
        // convert to physical pixels
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        // The window no longer exists so exit the application.
        return Err("Window no longer exists".to_string());
    };
    Ok((events_loop, surface, dimensions))
}


fn main() {
    // Uses config-rs to parse a .toml file at assets/settings.toml
    let mut settings = Config::new();
    settings
        .merge(config::File::with_name("assets/settings"))
        .expect("Failed to read settings file!");
    let settings_width = settings.get::<u32>("window.width").unwrap_or(800);
    let settings_height = settings.get::<u32>("window.height").unwrap_or(600);
    let settings_title = settings.get_str("window.title").unwrap_or("Please enter a window title!".to_string());

    let settings_fov = settings.get::<f32>("camera.fov").unwrap_or(75.0);
    let settings_near_cutoff = settings.get::<f32>("camera.near").unwrap_or(0.1);
    let settings_far_cutoff = settings.get::<f32>("camera.far").unwrap_or(100.0);


    // The first step of any Vulkan program is to create an instance.
    let instance = create_instance();

    // If we let this debug_callback binding fall out of scope then it will stop providing events
    #[cfg(debug_assertions)]
    let _debug_callback = create_debug_callback(instance.clone()).unwrap();

    // TODO: check for best option of physical device
    // For the sake of the example we are just going to use the first device.
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    // We first need to create the window. The events_loop contains the window events.
    let (mut events_loop, surface, mut dimensions) =
        create_window(instance.clone(), settings_width, settings_height, settings_title).unwrap();
    let window = surface.window();

    // The next step is to choose which GPU queue will execute our draw commands.
    // TODO: use one graphics and one transfer queue
    let queue_family = physical.queue_families().find(|&q| {
        // We take the first queue that supports drawing to our window.
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();


    // The list of created queues is returned by the function alongside with the device.
    let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
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

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Immediate,
            true,
            None,
        )
    }.unwrap();


    let vertices = VERTICES.iter().cloned();
    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();

    let normals = NORMALS.iter().cloned();
    let normals_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), normals).unwrap();

    let indices = INDICES.iter().cloned();
    let index_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), indices).unwrap();

    let uniform_buffer_pool = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

    // The next step is to create the shaders.
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    // -------------------------------------------------------------------------------------------
    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.
    // -------------------------------------------------------------------------------------------

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
                // This can be used for multisampling
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16Unorm,
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());

    let (mut pipeline, mut framebuffers) =
        window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

    // -------------------------------------------------------------------------------------------
    // Initialization is finally finished!
    // -------------------------------------------------------------------------------------------

    // In some situations, the swapchain will become invalid by itself.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<GpuFuture>;

    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
    // to correct for the vulkan clip space Y axis being the other way
    let clip = Matrix4::new(1.0, 0.0, 0.0, 0.0,
                            0.0, -1.0, 0.0, 0.0,
                            0.0, 0.0, 0.5, 0.0,
                            0.0, 0.0, 0.5, 1.0);
    let projection = clip * cgmath::perspective(
        Deg(settings_fov),
        aspect_ratio,
        settings_near_cutoff,
        settings_far_cutoff,
    );

    let model_teapot_1 = {
        let translation = Matrix4::from_translation(Vector3::new(1.5, 1.0, 0.0));
        let scale = Matrix4::from_nonuniform_scale(1.0, 2.0, 1.0);
        translation * scale * Matrix4::from_scale(0.01)
    };

    let model_teapot_2 = {
        let translation = Matrix4::from_translation(Vector3::new(-1.5, -1.0, 0.0));
        let rotation = Matrix4::from_angle_z(Deg(45.0));
        translation * rotation * Matrix4::from_scale(0.01)
    };

    let mut camera = ArcballCamera::new(
        6.0,
        Deg(90.0),
        0.0,
    );

    struct Ubo {
        projection: Matrix4<f32>,
    }

    let ubo = Ubo {
        projection
    };

    impl Ubo {
        fn get_uniform_data(&self, model_mat: Matrix4<f32>, view_mat: Matrix4<f32>) -> vs::ty::Data {
            vs::ty::Data {
                world: model_mat.into(),
                view: view_mat.into(),
                proj: self.projection.into(),
            }
        }
    }


    let (uniform_buffer_subbuffer_1, uniform_buffer_subbuffer_2) = {
        // phi = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;

        let view_mat = camera.generate_view_mat();

        let buf1 = {
            let uniform_data = ubo.get_uniform_data(model_teapot_1, view_mat);
            uniform_buffer_pool.next(uniform_data).unwrap()
        };

        let buf2 = {
            let uniform_data = ubo.get_uniform_data(model_teapot_2, view_mat);
            uniform_buffer_pool.next(uniform_data).unwrap()
        };
        (buf1, buf2)
    };


    let set_1 = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
        .add_buffer(uniform_buffer_subbuffer_1.clone()).unwrap()
        .build().unwrap()
    );

    let set_2 = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
        .add_buffer(uniform_buffer_subbuffer_2.clone()).unwrap()
        .build().unwrap()
    );



    loop {
        let now = Instant::now();
        // Handling the window events in order to close the program when the user wants to close
        // it.
        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent { event, .. } => {
                    if !camera.use_window_event(&event) {
                        match event {
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
                        }
                    }
                },
                _ => (),
            }
        });
        if done { return; }

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        previous_frame_end.cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        if recreate_swapchain {
            // Get the new dimensions of the window.
            dimensions = if let Some(dimensions) = window.get_inner_size() {
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

            // Because framebuffers contains an Arc on the old swapchain, and the pipeline contains
            // fixed viewport dimensions, we need to recreate them as well.
            let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                device.clone(),
                &vs,
                &fs,
                &new_swapchain_images,
                render_pass.clone(),
            );
            pipeline = new_pipeline;
            framebuffers = new_framebuffers;

            recreate_swapchain = false;
        }


        // Before we can draw on the output, we have to *acquire* an image from the swapchain.
        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            }
            Err(err) => panic!("{:?}", err)
        };

        let color_1 = Vector4::new(1.0, 0.0, 0.0, 1.0);
        let color_2 = Vector4::new(0.0, 0.0, 1.0, 1.0);

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec![
            [1.0, 1.0, 1.0, 1.0].into(),
            1f32.into()
        ];

        let view_mat = camera.generate_view_mat();

        // In order to draw, we have to build a *command buffer*.
        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family(),
        ).unwrap()
            // TODO: think about the best way to do uniforms and push constants
            // TODO: especially use a dedicated transfer queue
            .update_buffer(
                uniform_buffer_subbuffer_1.clone(),
                ubo.get_uniform_data(model_teapot_1, view_mat)
            ).unwrap()

            .update_buffer(
                uniform_buffer_subbuffer_2.clone(),
                ubo.get_uniform_data(model_teapot_2, view_mat)
            ).unwrap()

            // Before we can draw, we have to *enter a render pass*.
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()

            // We are now inside the first subpass of the render pass. We add a draw command.
            .draw_indexed(
                pipeline.clone(),
                &DynamicState::none(),
                vec!(vertex_buffer.clone(), normals_buffer.clone()),
                index_buffer.clone(),
                set_1.clone(),
                fs::ty::PushData { color: color_1.into() },
            ).unwrap()

            .draw_indexed(
                pipeline.clone(),
                &DynamicState::none(),
                vec!(vertex_buffer.clone(), normals_buffer.clone()),
                index_buffer.clone(),
                set_2.clone(),
                fs::ty::PushData { color: color_2.into() },
            ).unwrap()

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

        let elapsed = now.elapsed();
        println!("{}", elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0);

    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
) -> (Arc<GraphicsPipelineAbstract + Send + Sync>, Vec<Arc<FramebufferAbstract + Send + Sync>>) {
    let dimensions = images[0].dimensions();

    let depth_buffer = AttachmentImage::transient(
        device.clone(),
        dimensions,
        Format::D16Unorm,
    ).unwrap();

    let framebuffers = images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>();

    // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
    // This allows the driver to optimize things, at the cost of slower window resizes.
    // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .viewports(iter::once(Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        }))
        .fragment_shader(fs.main_entry_point(), ())
        .depth_stencil_simple_depth()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    (pipeline, framebuffers)
}