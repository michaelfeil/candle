use candle::{Result, Tensor};
#[derive(Debug)]
struct PreciseRope;
impl candle::CustomOp3 for PreciseRope {
    fn name(&self) -> &'static str {
        "gemma4-vision-rope"
    }
    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &candle::Layout,
        _: &candle::CpuStorage,
        _: &candle::Layout,
        _: &candle::CpuStorage,
        _: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("Gemma4 fused vision rotary embeddings require CUDA")
    }
    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        lx: &candle::Layout,
        c: &candle::CudaStorage,
        lc: &candle::Layout,
        s: &candle::CudaStorage,
        ls: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{CudaStorageSlice, WrapErr};
        fn run<T: DeviceRepr + candle::WithDType>(
            x: &CudaSlice<T>,
            lx: &candle::Layout,
            c: &CudaSlice<T>,
            lc: &candle::Layout,
            s: &CudaSlice<T>,
            ls: &candle::Layout,
            dev: &candle::CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let (_, h, t, d) = lx.shape().dims4()?;
            let count = u32::try_from(lx.shape().elem_count()).map_err(candle::Error::wrap)?;
            let total = u64::from(count);
            let h = u32::try_from(h).map_err(candle::Error::wrap)?;
            let t = u32::try_from(t).map_err(candle::Error::wrap)?;
            let d = u32::try_from(d).map_err(candle::Error::wrap)?;
            let stride = lx.stride();
            let sb = stride[0] as u64;
            let sh = stride[1] as u64;
            let st = stride[2] as u64;
            let sd = stride[3] as u64;
            let xs = x.slice(lx.start_offset()..);
            let cs = c.slice(lc.start_offset()..);
            let ss = s.slice(ls.start_offset()..);
            let name = candle::cuda_backend::kernel_name::<T>("gemma4_rope2d");
            let func = dev.get_or_load_func(&name, &candle::cuda_backend::kernels::REDUCE)?;
            let out = unsafe { dev.alloc::<T>(total as usize)? };
            let mut builder = func.builder();
            builder.arg(&xs);
            builder.arg(&cs);
            builder.arg(&ss);
            builder.arg(&out);
            candle::builder_arg!(builder, total, h, t, d, sb, sh, st, sd);
            unsafe { builder.launch(LaunchConfig::for_num_elems(count)) }.w()?;
            Ok(out)
        }
        let dev = x.device();
        let slice = match (&x.slice, &c.slice, &s.slice) {
            (CudaStorageSlice::BF16(x), CudaStorageSlice::BF16(c), CudaStorageSlice::BF16(s)) => {
                CudaStorageSlice::BF16(run(x, lx, c, lc, s, ls, dev)?)
            }
            (CudaStorageSlice::F16(x), CudaStorageSlice::F16(c), CudaStorageSlice::F16(s)) => {
                CudaStorageSlice::F16(run(x, lx, c, lc, s, ls, dev)?)
            }
            (CudaStorageSlice::F32(x), CudaStorageSlice::F32(c), CudaStorageSlice::F32(s)) => {
                CudaStorageSlice::F32(run(x, lx, c, lc, s, ls, dev)?)
            }
            _ => candle::bail!("dtype mismatch"),
        };
        Ok((
            candle::CudaStorage {
                slice,
                device: dev.clone(),
            },
            lx.shape().clone(),
        ))
    }
}
pub(super) fn apply(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, _, t, d) = xs.dims4()?;
    if xs.elem_count() == 0
        || d % 4 != 0
        || cos.dims3()? != (b, t, d)
        || sin.dims3()? != (b, t, d)
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || cos.dtype() != xs.dtype()
        || sin.dtype() != xs.dtype()
        || !xs.device().same_device(cos.device())
        || !xs.device().same_device(sin.device())
    {
        candle::bail!("Invalid Gemma4 vision rotary inputs")
    }
    xs.apply_op3_no_bwd(cos, sin, &PreciseRope)
}
