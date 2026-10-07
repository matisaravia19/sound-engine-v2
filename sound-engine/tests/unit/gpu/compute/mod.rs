use super::*;

impl ComputeContext {
    /// Returns the live pipeline count for lifetime regression tests.
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }
}
