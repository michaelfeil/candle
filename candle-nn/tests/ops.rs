#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, test_utils::to_vec3_round, Device, IndexOp, Result, Tensor};

fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = candle_nn::ops::softmax(&tensor.log()?, 0)?;
    let t1 = candle_nn::ops::softmax(&tensor.log()?, 1)?;
    let t2 = candle_nn::ops::softmax(&tensor.log()?, 2)?;
    assert_eq!(
        to_vec3_round(&t0, 4)?,
        &[
            // 3/5, 1/2, 4/11
            [[0.6, 0.5, 0.3636], [0.1111, 0.7143, 0.5294]],
            // 2/5, 1/2, 7/11
            [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t1, 4)?,
        &[
            // 3/4, 1/6, 4/13
            [[0.75, 0.1667, 0.3077], [0.25, 0.8333, 0.6923]],
            // 2/10, 1/3, 7/15
            [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    let t2 = candle_nn::ops::softmax_last_dim(&tensor.log()?)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    Ok(())
}

fn rms_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn rms_norm_f32_math(device: &Device) -> Result<()> {
    use candle::{DType, D};
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for cols in [3, 17, 72, 128, 512, 1024, 1152, 4096] {
            let values = (0..7 * cols)
                .map(|i| {
                    if i < 2 * cols {
                        0.
                    } else if i / cols == 2 {
                        // Sum 17 distinguishes rounded reciprocal multiplication from
                        // float32 division at widths 72 and 1152.
                        if i % cols < 17 {
                            1.
                        } else {
                            0.
                        }
                    } else {
                        ((i * 31 % 199) as f32 - 99.) / 33.
                    }
                })
                .collect::<Vec<_>>();
            // Exercise storage offsets as well as zero-input rows.
            let tensor = Tensor::from_vec(values, (7, cols), device)?
                .to_dtype(dtype)?
                .narrow(0, 1, 5)?;
            let weights = (0..cols)
                .map(|i| 0.85 + (i % 11) as f32 * 0.037)
                .collect::<Vec<_>>();
            let alpha = Tensor::from_vec(weights, cols, device)?.to_dtype(dtype)?;
            let x = tensor.to_dtype(DType::F32)?;
            let variance = (x.sqr()?.sum_keepdim(D::Minus1)? / cols as f64)?;
            let expected = x
                .broadcast_div(&(variance + 1e-6)?.sqrt()?)?
                .broadcast_mul(&alpha.to_dtype(DType::F32)?)?
                .to_dtype(dtype)?;
            let actual = candle_nn::ops::rms_norm_f32(&tensor, &alpha, 1e-6)?;
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                "{dtype:?} width {cols}"
            );
        }
    }
    for shape in [(0, 72), (0, 1152)] {
        let empty = Tensor::zeros(shape, DType::F32, device)?;
        let alpha = Tensor::ones(shape.1, DType::F32, device)?;
        assert_eq!(
            candle_nn::ops::rms_norm_f32(&empty, &alpha, 1e-6)?.dims(),
            empty.dims()
        );
    }
    assert!(candle_nn::ops::rms_norm_f32(
        &Tensor::zeros((1, 0), candle::DType::F32, device)?,
        &Tensor::zeros(0, candle::DType::F32, device)?,
        1e-6
    )
    .is_err());
    Ok(())
}

fn rms_norm_f32_gradients(device: &Device) -> Result<()> {
    use candle::{DType, Var, D};
    for (track_x, track_alpha) in [(true, false), (false, true), (true, true)] {
        let x_var = Var::from_vec(vec![1f32, -2., 3., 4., 5., -6.], (2, 3), device)?;
        let alpha_var = Var::from_vec(vec![0.8f32, 1.2, 0.9], 3, device)?;
        let x = if track_x {
            x_var.as_tensor().clone()
        } else {
            x_var.detach()
        };
        let alpha = if track_alpha {
            alpha_var.as_tensor().clone()
        } else {
            alpha_var.detach()
        };
        let variance = (x.sqr()?.sum_keepdim(D::Minus1)? / 3.)?;
        let expected = x
            .broadcast_div(&(variance + 1e-6)?.sqrt()?)?
            .broadcast_mul(&alpha)?;
        let actual = candle_nn::ops::rms_norm_f32(&x, &alpha, 1e-6)?;
        let expected_grads = expected.sum_all()?.backward()?;
        let actual_grads = actual.sum_all()?.backward()?;
        for (tracked, var) in [(track_x, &x_var), (track_alpha, &alpha_var)] {
            if tracked {
                let expected_grad = expected_grads.get(var).expect("reference gradient");
                let actual_grad = actual_grads.get(var).expect("gradient must be preserved");
                assert_eq!(
                    actual_grad
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected_grad
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
        }
    }
    Ok(())
}

fn rms_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(to_vec3_round(&t, 2)?, to_vec3_round(&t2, 2)?);
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn rms_norm_large_magnitude(device: &Device) -> Result<()> {
    let (rows, hidden) = (4usize, 6912usize);
    let data: Vec<f32> = (0..rows * hidden)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * ((i as f32 * 0.17).sin().abs() * 7e9)
        })
        .collect();
    let tensor = Tensor::from_vec(data, (rows, hidden), device)?;
    let alpha = Tensor::ones(hidden, candle::DType::F32, device)?;

    let fused = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let slow = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;

    let fused_v = fused.flatten_all()?.to_vec1::<f32>()?;
    let slow_v = slow.flatten_all()?.to_vec1::<f32>()?;
    for &v in &fused_v {
        assert!(
            v.is_finite(),
            "rms_norm produced a non-finite value for large-magnitude input"
        );
    }
    for &v in &slow_v {
        assert!(
            v.is_finite(),
            "rms_norm_slow produced a non-finite value for large-magnitude input"
        );
    }
    let diff = fused_v
        .iter()
        .zip(slow_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < 5e-3,
        "rms_norm and rms_norm_slow disagree: max |Δ| = {diff}"
    );
    Ok(())
}

fn layer_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn layer_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

#[test]
fn softmax_numerical_stability() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = Tensor::new(&[1234f32, 0.], dev)?;
    let softmax = candle_nn::ops::softmax(&xs, 0)?;
    assert_eq!(softmax.to_vec1::<f32>()?, &[1f32, 0.]);
    Ok(())
}

fn ropei(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope_i(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope_thd(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &cos, &sin)?.transpose(1, 2)?
    };
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(0..1)?, &cos, &sin)?
    };
    let rope2 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(1..2)?, &cos2, &sin2)?
    };

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &both_cos, &both_sin)?
    };
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn sigmoid(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let s1 = candle_nn::ops::sigmoid(&tensor)?;
    let s2 = (1. / (1. + tensor.neg()?.exp()?)?)?;
    let diff = (s1 - s2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

test_device!(ropei, ropei_cpu, ropei_gpu, ropei_metal);
test_device!(rope, rope_cpu, rope_gpu, rope_metal);
test_device!(rope_thd, rope_thd_cpu, rope_thd_gpu, rope_thd_metal);
test_device!(softmax, softmax_cpu, softmax_gpu, softmax_metal);
test_device!(rms_norm, rms_norm_cpu, rms_norm_gpu, rms_norm_metal);
test_device!(
    rms_norm_f32_math,
    rms_norm_f32_math_cpu,
    rms_norm_f32_math_gpu,
    rms_norm_f32_math_metal
);
test_device!(rms_norml, rms_norml_cpu, rms_norml_gpu, rms_norml_metal);
test_device!(
    rms_norm_large_magnitude,
    rms_norm_large_magnitude_cpu,
    rms_norm_large_magnitude_gpu,
    rms_norm_large_magnitude_metal
);
test_device!(layer_norm, ln_cpu, ln_gpu, ln_metal);
test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal);
test_device!(sigmoid, sigmoid_cpu, sigmoid_gpu, sigmoid_metal);

test_device!(
    rms_norm_f32_gradients,
    rms_norm_f32_gradients_cpu,
    rms_norm_f32_gradients_gpu,
    rms_norm_f32_gradients_metal
);
