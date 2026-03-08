/*
 * Copyright 2026 Sebastian Tilders - All rights reserved
 */

//! Bridge between Rayon's CPU-bound thread pool and async Tokio code.
//!
//! [`schedule_rayon_task`] spawns a fallible, panic-safe closure on a [`rayon::ThreadPool`]
//! and returns a [`TaskHandle`] that can be `.await`ed inside a Tokio runtime. This lets
//! blocking or compute-heavy work run on Rayon's work-stealing threads without blocking
//! the async executor.
//!
//! # Error handling
//!
//! All failure modes are captured in [`TaskExecutionError`]:
//! - [`TaskExecutionError::TaskError`] — the closure returned `Err(E)`.
//! - [`TaskExecutionError::TaskPanic`] — the closure panicked (panic value is stringified).
//! - [`TaskExecutionError::JoinError`] — the Tokio wrapper task was cancelled or panicked.
//! - [`TaskExecutionError::ResultSendError`] — internal channel error (should not occur in normal use).

use std::any::Any;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::panic;
use std::panic::UnwindSafe;
use std::pin::Pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll};
use tokio::sync;
use tokio::task::{JoinError, JoinHandle};

/// Schedules a fallible closure on a Rayon thread pool and returns a [`TaskHandle`] that
/// can be awaited from async code.
///
/// The task is spawned with FIFO ordering via [`rayon::ThreadPool::spawn_fifo`]. Panics
/// inside `func` are caught and surfaced as [`TaskExecutionError::TaskPanic`].
///
/// # Parameters
/// - `thread_pool` — the Rayon thread pool to run `func` on.
/// - `func` — a `FnOnce` closure returning `Result<R, E>`.
///
/// # Returns
/// A [`TaskHandle`] whose output is `Result<R, `[`TaskExecutionError<E>`]`>`.
#[inline]
pub fn schedule_rayon_task<R, E>(
    thread_pool: Arc<rayon::ThreadPool>,
    func: impl FnOnce() -> Result<R, E> + Send + UnwindSafe + 'static,
) -> TaskHandle<R, E>
where
    R: Send + 'static,
    E: Display + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let notify = Arc::new(sync::Notify::new());
    let thread_notify = notify.clone();

    thread_pool.spawn_fifo(move || {
        let r = panic::catch_unwind(func);
        let _ = tx.send(r);
        thread_notify.notify_one();
    });

    let task = async move {
        notify.notified().await;
        let res = rx.try_recv()?;
        match res {
            Ok(r) => r.map_err(|e| TaskExecutionError::TaskError(TaskError(e))),
            Err(err) => Err(translate_string_panics(err)),
        }
    };

    TaskHandle(tokio::spawn(task))
}

/// A handle to a rayon task scheduled via [`schedule_rayon_task`].
///
/// Implements [`Future`], resolving to `Result<R, `[`TaskExecutionError<E>`]`>`.
/// Dropping the handle does not cancel the underlying rayon task.
#[repr(transparent)]
pub struct TaskHandle<R, E>(JoinHandle<Result<R, TaskExecutionError<E>>>);

impl<R, E> Future for TaskHandle<R, E> {
    type Output = Result<R, TaskExecutionError<E>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match std::pin::pin!(&mut self.0).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(res) => match res {
                Ok(res) => Poll::Ready(res),
                Err(e) => Poll::Ready(Err(TaskExecutionError::JoinError(e))),
            },
        }
    }
}

/// Errors that can occur when executing a task scheduled via [`schedule_rayon_task`].
pub enum TaskExecutionError<E> {
    /// The task closure returned an `Err(E)`.
    TaskError(TaskError<E>),
    /// The result channel was in an unexpected state (should not happen under normal use).
    ResultSendError(mpsc::TryRecvError),
    /// The Tokio [`JoinHandle`] failed, e.g. due to task cancellation.
    JoinError(JoinError),
    /// The task closure panicked; contains a best-effort string representation of the panic value.
    TaskPanic(String),
}

fn translate_string_panics<E: Display>(e: Box<dyn Any>) -> TaskExecutionError<E> {
    if let Some(str_slice_err) = e.downcast_ref::<&str>() {
        return TaskExecutionError::TaskPanic(str_slice_err.to_string());
    } else if let Some(str_err) = e.downcast_ref::<String>() {
        return TaskExecutionError::TaskPanic(str_err.clone());
    }
    TaskExecutionError::TaskPanic(format!("Unknown panic: {:?}", e))
}

impl<E> Display for TaskExecutionError<E>
where
    E: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskExecutionError::TaskError(err) => write!(f, "TaskError: {err}"),
            TaskExecutionError::ResultSendError(err) => write!(f, "ResultSendError: {err}"),
            TaskExecutionError::JoinError(err) => write!(f, "JoinError: {err}"),
            TaskExecutionError::TaskPanic(err) => write!(f, "TaskPanic: {err}"),
        }
    }
}

impl<E> Debug for TaskExecutionError<E>
where
    E: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskExecutionError::TaskError(err) => write!(f, "TaskError: {err:?}"),
            TaskExecutionError::ResultSendError(err) => write!(f, "ResultSendError: {err:?}"),
            TaskExecutionError::JoinError(err) => write!(f, "JoinError: {err:?}"),
            TaskExecutionError::TaskPanic(err) => write!(f, "TaskPanic: {err:?}"),
        }
    }
}

impl<E: Debug + Display + 'static> Error for TaskExecutionError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TaskExecutionError::TaskError(err) => Some(*Box::new(err as &dyn Error)),
            TaskExecutionError::ResultSendError(err) => Some(err),
            TaskExecutionError::JoinError(err) => Some(err),
            TaskExecutionError::TaskPanic(_) => None,
        }
    }
}

impl<E> From<mpsc::TryRecvError> for TaskExecutionError<E> {
    fn from(err: mpsc::TryRecvError) -> Self {
        TaskExecutionError::ResultSendError(err)
    }
}

impl<E> From<JoinError> for TaskExecutionError<E> {
    fn from(value: JoinError) -> Self {
        TaskExecutionError::JoinError(value)
    }
}

/// Wraps the error value `E` returned by a task closure.
#[derive(Debug)]
pub struct TaskError<E>(E);

impl<E: Display> Display for TaskError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl<E: Debug + Display> Error for TaskError<E> {}
